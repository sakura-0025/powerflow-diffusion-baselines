"""Common statistical, physical and speed evaluation for all baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from data.dataset import load_archive
from model.physics import ac_power_balance, line_apparent_power, physics_from_archive


CHANNELS = ("p_injection_pu", "q_injection_pu", "vm_pu", "theta_rad")


def _mmd(real: np.ndarray, generated: np.ndarray, max_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    count = min(max_samples, len(real), len(generated))
    real = real[rng.choice(len(real), count, replace=False)].reshape(count, -1)
    generated = generated[rng.choice(len(generated), count, replace=False)].reshape(count, -1)
    location = real.mean(axis=0)
    scale = np.maximum(real.std(axis=0), 1.0e-6)
    real = (real - location) / scale
    generated = (generated - location) / scale
    probe = real[: min(500, count)]
    distance = np.sum((probe[:, None] - probe[None, :]) ** 2, axis=-1)
    positive = distance[distance > 0]
    bandwidth = max(float(np.median(positive)) if len(positive) else 1.0, 1.0e-12)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        squared = np.sum((left[:, None] - right[None, :]) ** 2, axis=-1)
        return np.exp(-squared / (2.0 * bandwidth))

    return max(
        float(kernel(real, real).mean() + kernel(generated, generated).mean() - 2 * kernel(real, generated).mean()),
        0.0,
    )


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    archive = load_archive(args.data)
    with np.load(args.generated, allow_pickle=False) as source:
        generated = source["generated_common"].astype(np.float64)
        sampling_metadata = json.loads(str(source["metadata_json"].item()))
    split = np.asarray(archive["split"], dtype=np.int8)
    test = np.asarray(archive["state_common"], dtype=np.float64)[split == 2]
    rng = np.random.default_rng(args.seed)
    reference = test[rng.choice(len(test), len(generated), replace=len(generated) > len(test))]
    if generated.shape != reference.shape:
        raise ValueError(f"Generated shape {generated.shape} does not match reference {reference.shape}.")
    if not np.isfinite(generated).all():
        raise ValueError("Generated samples contain NaN or Inf.")

    wasserstein: dict[str, float] = {}
    mean_error: dict[str, float] = {}
    std_ratio: dict[str, float] = {}
    for channel, name in enumerate(CHANNELS):
        per_bus = [
            wasserstein_distance(reference[:, bus, channel], generated[:, bus, channel])
            for bus in range(reference.shape[1])
        ]
        wasserstein[name] = float(np.mean(per_bus))
        mean_error[name] = float(
            np.mean(np.abs(reference[..., channel].mean(axis=0) - generated[..., channel].mean(axis=0)))
        )
        std_ratio[name] = float(
            generated[..., channel].std() / max(reference[..., channel].std(), 1.0e-12)
        )

    real_flat = reference.reshape(len(reference), -1)
    generated_flat = generated.reshape(len(generated), -1)
    valid = (real_flat.std(axis=0) > 1.0e-10) & (generated_flat.std(axis=0) > 1.0e-10)
    correlation_error: float | None = None
    if int(valid.sum()) >= 2:
        real_corr = np.corrcoef(real_flat[:, valid], rowvar=False)
        generated_corr = np.corrcoef(generated_flat[:, valid], rowvar=False)
        correlation_error = float(
            np.linalg.norm(real_corr - generated_corr, ord="fro") / real_corr.shape[0]
        )

    grid = physics_from_archive(archive)
    state = torch.from_numpy(generated.astype(np.float32))
    with torch.no_grad():
        residual = ac_power_balance(state, grid).abs().numpy()
        from_flow, to_flow = line_apparent_power(state, grid)
    sample_mean = residual.mean(axis=(1, 2))
    sample_max = residual.max(axis=(1, 2))
    voltage = generated[..., 2]
    lower = np.asarray(archive["common_lower"])
    upper = np.asarray(archive["common_upper"])
    voltage_violation = np.any(
        (voltage < lower[:, 2]) | (voltage > upper[:, 2]), axis=1
    )
    limits = np.asarray(archive["branch_rate_pu"])
    finite = np.isfinite(limits)
    thermal_violation_rate = 0.0
    if finite.any():
        violation = np.any(
            (from_flow.numpy()[:, finite] > limits[finite])
            | (to_flow.numpy()[:, finite] > limits[finite]),
            axis=1,
        )
        thermal_violation_rate = float(violation.mean())

    return {
        "dataset": str(args.data.resolve()),
        "generated": str(args.generated.resolve()),
        "method": sampling_metadata["method"],
        "sample_count": len(generated),
        "wasserstein_per_channel": wasserstein,
        "absolute_mean_error_per_channel": mean_error,
        "generated_to_real_std_ratio": std_ratio,
        "mmd_rbf_squared": _mmd(reference, generated, args.metric_samples, args.seed),
        "correlation_frobenius_error": correlation_error,
        "physics": {
            "mean_absolute_residual_pu": float(residual.mean()),
            "p95_absolute_residual_pu": float(np.quantile(residual, 0.95)),
            "p99_absolute_residual_pu": float(np.quantile(residual, 0.99)),
            "sample_mean_residual_p95_pu": float(np.quantile(sample_mean, 0.95)),
            "sample_max_residual_median_pu": float(np.median(sample_max)),
            "sample_max_residual_p95_pu": float(np.quantile(sample_max, 0.95)),
            "feasible_rate_by_tolerance": {
                f"{value:g}": float(np.mean(sample_max <= value))
                for value in (1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2)
            },
        },
        "voltage_violation_rate": float(voltage_violation.mean()),
        "thermal_violation_rate": thermal_violation_rate,
        "sampling": sampling_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
