"""Audit a generated benchmark archive before model training.

The audit is deliberately independent of the model code: it verifies shapes,
splits, finite values and the AC nodal-balance residual saved in the archive.
Run it on the server immediately after every large dataset build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.dataset import load_archive
from model.physics import ac_power_balance, line_apparent_power, physics_from_archive


def audit(path: Path, residual_tolerance: float) -> dict[str, object]:
    archive = load_archive(path)
    common = np.asarray(archive["state_common"], dtype=np.float32)
    wang = np.asarray(archive["state_wang"], dtype=np.float32)
    split = np.asarray(archive["split"], dtype=np.int8)
    metadata = archive["metadata"]
    load_scale = archive.get("load_scale")
    system_load_scale = archive.get("system_load_scale")
    rejected_system_load_scale = archive.get("rejected_system_load_scale")

    errors: list[str] = []
    if common.ndim != 3 or common.shape[-1] != 4:
        errors.append(f"state_common must have shape [S,B,4], got {common.shape}")
    if wang.ndim != 2 or len(wang) != len(common):
        errors.append(f"state_wang must have shape [S,D], got {wang.shape}")
    if split.shape != (len(common),):
        errors.append(f"split must have shape ({len(common)},), got {split.shape}")
    if not set(np.unique(split)).issubset({0, 1, 2}):
        errors.append("split contains a code outside {0,1,2}")
    if any(int(np.sum(split == code)) == 0 for code in (0, 1, 2)):
        errors.append("train, validation and test must all be non-empty")
    if not np.isfinite(common).all() or not np.isfinite(wang).all():
        errors.append("state arrays contain NaN or Inf")

    load_scale_report: dict[str, object] | None = None
    if load_scale is not None and system_load_scale is not None:
        load_scale_array = np.asarray(load_scale, dtype=np.float64)
        system_scale_array = np.asarray(system_load_scale, dtype=np.float64)
        expected_low, expected_high = (
            (0.8, 1.0) if metadata["protocol"] == "hoseinpour" else (0.8, 1.2)
        )
        if load_scale_array.shape[0] != len(common):
            errors.append("load_scale sample dimension does not match state arrays")
        if (
            np.any(load_scale_array < expected_low - 1.0e-6)
            or np.any(load_scale_array > expected_high + 1.0e-6)
        ):
            errors.append("load_scale contains a value outside the protocol bounds")

        rejected = np.asarray(
            rejected_system_load_scale
            if rejected_system_load_scale is not None
            else np.empty(0),
            dtype=np.float64,
        )

        combined = np.concatenate([system_scale_array, rejected])
        bin_edges = np.linspace(combined.min(), combined.max(), 11)
        # Include the maximum in the final half-open histogram interval.
        bin_edges[-1] = np.nextafter(bin_edges[-1], np.inf)
        accepted_histogram, _ = np.histogram(system_scale_array, bins=bin_edges)
        rejected_histogram, _ = np.histogram(rejected, bins=bin_edges)
        acceptance_by_bin: list[dict[str, float | int | None]] = []
        for index, (accepted_count, rejected_count) in enumerate(
            zip(accepted_histogram, rejected_histogram, strict=True)
        ):
            attempted_count = int(accepted_count + rejected_count)
            acceptance_by_bin.append(
                {
                    "lower": float(bin_edges[index]),
                    "upper": float(bin_edges[index + 1]),
                    "attempted": attempted_count,
                    "accepted": int(accepted_count),
                    "rejected": int(rejected_count),
                    "acceptance_rate": (
                        float(accepted_count / attempted_count)
                        if attempted_count
                        else None
                    ),
                }
            )

        def distribution(values: np.ndarray) -> dict[str, float] | None:
            if len(values) == 0:
                return None
            return {
                "minimum": float(values.min()),
                "mean": float(values.mean()),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(values.max()),
            }

        load_scale_report = {
            "protocol_bounds": [expected_low, expected_high],
            "accepted_system_load_scale": distribution(system_scale_array),
            "rejected_system_load_scale": distribution(rejected),
            "rejected_count": len(rejected),
            "acceptance_by_system_load_bin": acceptance_by_bin,
        }

    # Chunk the physics check so the 118-bus full dataset does not require a
    # large temporary tensor. Residual units are per unit on metadata baseMVA.
    grid = physics_from_archive(archive)
    residual_sum = 0.0
    residual_count = 0
    residual_max = 0.0
    source_thermal_violation_count = 0
    for start in range(0, len(common), 4096):
        state = torch.from_numpy(common[start : start + 4096])
        with torch.no_grad():
            residual = ac_power_balance(state, grid).abs()
            from_flow, to_flow = line_apparent_power(state, grid)
        residual_sum += float(residual.sum())
        residual_count += residual.numel()
        residual_max = max(residual_max, float(residual.max()))
        finite_limits = torch.isfinite(grid.branch_rate_pu)
        if bool(finite_limits.any()):
            thermal_violation = torch.any(
                (from_flow[:, finite_limits] > grid.branch_rate_pu[finite_limits] + 1.0e-5)
                | (to_flow[:, finite_limits] > grid.branch_rate_pu[finite_limits] + 1.0e-5),
                dim=1,
            )
            source_thermal_violation_count += int(thermal_violation.sum())
    residual_mean = residual_sum / max(residual_count, 1)
    if residual_max > residual_tolerance:
        errors.append(
            f"maximum AC residual {residual_max:.6g} exceeds {residual_tolerance:.6g} pu"
        )

    voltage = common[..., 2]
    lower = np.asarray(archive["common_lower"], dtype=np.float64)[:, 2]
    upper = np.asarray(archive["common_upper"], dtype=np.float64)[:, 2]
    source_voltage_violation = np.any(
        (voltage < lower - 1.0e-6) | (voltage > upper + 1.0e-6), axis=1
    )
    source_voltage_violation_count = int(source_voltage_violation.sum())
    if source_voltage_violation_count:
        errors.append(
            f"{source_voltage_violation_count} source samples violate stored voltage bounds"
        )
    if source_thermal_violation_count:
        errors.append(
            f"{source_thermal_violation_count} source samples violate stored branch limits"
        )

    return {
        "status": "PASS" if not errors else "FAIL",
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata": metadata,
        "state_common_shape": list(common.shape),
        "state_wang_shape": list(wang.shape),
        "split_counts": {
            "train": int(np.sum(split == 0)),
            "validation": int(np.sum(split == 1)),
            "test": int(np.sum(split == 2)),
        },
        "ac_residual_mean_pu": residual_mean,
        "ac_residual_max_pu": residual_max,
        "residual_tolerance_pu": residual_tolerance,
        "source_voltage_violation_count": source_voltage_violation_count,
        "source_thermal_violation_count": source_thermal_violation_count,
        "load_scale_audit": load_scale_report,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.data, args.residual_tolerance)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
