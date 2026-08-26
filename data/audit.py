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
from model.physics import ac_power_balance, physics_from_archive


def audit(path: Path, residual_tolerance: float) -> dict[str, object]:
    archive = load_archive(path)
    common = np.asarray(archive["state_common"], dtype=np.float32)
    wang = np.asarray(archive["state_wang"], dtype=np.float32)
    split = np.asarray(archive["split"], dtype=np.int8)
    metadata = archive["metadata"]

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

    # Chunk the physics check so the 118-bus full dataset does not require a
    # large temporary tensor. Residual units are per unit on metadata baseMVA.
    grid = physics_from_archive(archive)
    residual_sum = 0.0
    residual_count = 0
    residual_max = 0.0
    for start in range(0, len(common), 4096):
        state = torch.from_numpy(common[start : start + 4096])
        with torch.no_grad():
            residual = ac_power_balance(state, grid).abs()
        residual_sum += float(residual.sum())
        residual_count += residual.numel()
        residual_max = max(residual_max, float(residual.max()))
    residual_mean = residual_sum / max(residual_count, 1)
    if residual_max > residual_tolerance:
        errors.append(
            f"maximum AC residual {residual_max:.6g} exceeds {residual_tolerance:.6g} pu"
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
