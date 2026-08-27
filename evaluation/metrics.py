"""Paper-aligned metrics for synthetic power-flow datasets.

The two reproduced papers do not use exactly the same evaluation protocol.
This module therefore keeps source-stated metrics and project-controlled
supplements in one report while recording the provenance of each metric.

Source-stated metrics
----------------------
* Wang et al. Eq. (7): mean magnitude of the complex nodal power imbalance.
* Hoseinpour--Dvorkin Eq. (33): type-1 Wasserstein distance between the joint
  real and synthetic distributions.
* Hoseinpour--Dvorkin Table II/Fig. 7: per-bus mean and standard deviation of
  signed active/reactive power mismatches.

The full empirical optimal-transport problem is cubic in the number of
samples.  We therefore report a deterministic equal-mass assignment on a
seeded subsample and name it explicitly as an approximation.  Marginal
Wasserstein distances, MMD, correlation error, quantiles, feasibility rates,
and runtime counters are project-controlled common metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from data.dataset import load_archive
from model.physics import ac_power_balance, line_apparent_power, physics_from_archive


CHANNELS = ("p_injection_pu", "q_injection_pu", "vm_pu", "theta_rad")
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


METRIC_PROVENANCE = {
    "joint_wasserstein1_normalized_subsampled": (
        "source-stated metric (Hoseinpour-Dvorkin Eq. 33); project approximation "
        "using seeded equal-mass optimal assignment on nonconstant features after "
        "train-set scaling"
    ),
    "mean_complex_power_imbalance_pu": "source-stated metric (Wang et al. Eq. 7)",
    "per_bus_power_mismatch": (
        "source-stated analysis (Hoseinpour-Dvorkin Table II and Fig. 7)"
    ),
    "marginal_wasserstein_per_channel": "project-controlled common supplement",
    "mmd_rbf_squared": "project-controlled common supplement",
    "correlation_frobenius_error": "project-controlled common supplement",
    "residual_quantiles_and_feasible_rates": "project-controlled common supplement",
    "bound_and_thermal_violation_rates": (
        "project-controlled implementation of the inequality constraints; p/q/theta "
        "bounds are train-split extrema and voltage/thermal limits come from the case"
    ),
    "sampling_speed_and_compute_counts": "project-controlled efficiency supplement",
}


def _seeded_indices(length: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Choose deterministic indices, with replacement only when unavoidable."""

    if length <= 0:
        raise ValueError("Cannot sample from an empty array.")
    return rng.choice(length, count, replace=count > length)


def _train_minmax(archive: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the common-state scaler fitted by ``train.py``."""

    split = np.asarray(archive["split"], dtype=np.int8)
    train = np.asarray(archive["state_common"], dtype=np.float64)[split == 0]
    if not len(train):
        raise ValueError("The archive has no training samples.")
    lower = train.min(axis=0)
    upper = train.max(axis=0)
    return lower, upper


def normalize_common(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Scale common states to [-1, 1], matching the common-model protocol."""

    # Constant coordinates (for example the slack-bus angle) are excluded from
    # joint transport below.  A unit span keeps this standalone helper finite.
    scale = np.where((upper - lower) > 1.0e-8, upper - lower, 1.0)
    return 2.0 * (values - lower) / scale - 1.0


def rbf_mmd_squared(
    real: np.ndarray,
    generated: np.ndarray,
    max_samples: int,
    seed: int,
) -> float:
    """Return a seeded RBF MMD estimate on standardized flattened states."""

    rng = np.random.default_rng(seed)
    count = min(max_samples, len(real), len(generated))
    if count < 2:
        raise ValueError("MMD requires at least two real and generated samples.")
    real = real[_seeded_indices(len(real), count, rng)].reshape(count, -1)
    generated = generated[_seeded_indices(len(generated), count, rng)].reshape(count, -1)
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
        float(
            kernel(real, real).mean()
            + kernel(generated, generated).mean()
            - 2.0 * kernel(real, generated).mean()
        ),
        0.0,
    )


def joint_wasserstein1_subsampled(
    real_normalized: np.ndarray,
    generated_normalized: np.ndarray,
    max_samples: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Approximate Hoseinpour Eq. (33) by optimal assignment on a subsample.

    For equally weighted empirical distributions with the same sample count,
    the discrete W1 problem reduces to a minimum-cost bipartite assignment.
    The returned value is the mean Euclidean assignment cost.
    """

    rng = np.random.default_rng(seed)
    count = min(max_samples, len(real_normalized), len(generated_normalized))
    if count < 1:
        raise ValueError("Joint Wasserstein distance requires non-empty arrays.")
    real = real_normalized[_seeded_indices(len(real_normalized), count, rng)].reshape(count, -1)
    generated = generated_normalized[
        _seeded_indices(len(generated_normalized), count, rng)
    ].reshape(count, -1)
    cost = cdist(real, generated, metric="euclidean")
    rows, columns = linear_sum_assignment(cost)
    return {
        "value": float(cost[rows, columns].mean()),
        "sample_count": int(count),
        "ground_metric": "euclidean",
        "state_scaling": "train_common_minmax_to_-1_1",
        "feature_count": int(real.shape[1]),
        "estimator": "equal_mass_linear_assignment_on_seeded_subsample",
    }


def _load_generated(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        generated = source["generated_common"].astype(np.float64)
        metadata = json.loads(str(source["metadata_json"].item()))
    if generated.ndim != 3 or generated.shape[-1] != 4:
        raise ValueError("generated_common must have shape [sample, bus, 4].")
    if not np.isfinite(generated).all():
        raise ValueError("Generated samples contain NaN or Inf.")
    return generated, metadata


def evaluate_generated(
    data_path: str | Path,
    generated_path: str | Path,
    *,
    split_name: str = "test",
    metric_samples: int = 1_000,
    transport_samples: int = 256,
    seed: int = 2026,
) -> dict[str, object]:
    """Evaluate one generated archive under a common seeded protocol."""

    if split_name not in SPLIT_CODES:
        raise ValueError(f"Unknown split {split_name!r}; choose from {tuple(SPLIT_CODES)}.")
    data_path = Path(data_path)
    generated_path = Path(generated_path)
    archive = load_archive(data_path)
    generated, sampling_metadata = _load_generated(generated_path)

    split = np.asarray(archive["split"], dtype=np.int8)
    pool = np.asarray(archive["state_common"], dtype=np.float64)[
        split == SPLIT_CODES[split_name]
    ]
    rng = np.random.default_rng(seed)
    reference = pool[_seeded_indices(len(pool), len(generated), rng)]
    if generated.shape != reference.shape:
        raise ValueError(
            f"Generated shape {generated.shape} does not match reference shape {reference.shape}."
        )

    marginal_wasserstein: dict[str, float] = {}
    marginal_wasserstein_per_bus: dict[str, list[float]] = {}
    mean_error: dict[str, float] = {}
    std_ratio: dict[str, float] = {}
    for channel, name in enumerate(CHANNELS):
        per_bus = np.asarray(
            [
                wasserstein_distance(reference[:, bus, channel], generated[:, bus, channel])
                for bus in range(reference.shape[1])
            ],
            dtype=np.float64,
        )
        marginal_wasserstein[name] = float(per_bus.mean())
        marginal_wasserstein_per_bus[name] = per_bus.tolist()
        mean_error[name] = float(
            np.mean(
                np.abs(
                    reference[..., channel].mean(axis=0)
                    - generated[..., channel].mean(axis=0)
                )
            )
        )
        std_ratio[name] = float(
            generated[..., channel].std()
            / max(reference[..., channel].std(), 1.0e-12)
        )

    real_flat = reference.reshape(len(reference), -1)
    generated_flat = generated.reshape(len(generated), -1)
    valid = (real_flat.std(axis=0) > 1.0e-10) & (
        generated_flat.std(axis=0) > 1.0e-10
    )
    correlation_error: float | None = None
    if int(valid.sum()) >= 2:
        real_corr = np.corrcoef(real_flat[:, valid], rowvar=False)
        generated_corr = np.corrcoef(generated_flat[:, valid], rowvar=False)
        correlation_error = float(
            np.linalg.norm(real_corr - generated_corr, ord="fro") / real_corr.shape[0]
        )

    train_lower, train_upper = _train_minmax(archive)
    train_span = (train_upper - train_lower).reshape(-1)
    nonconstant = train_span > 1.0e-8
    if not np.any(nonconstant):
        raise ValueError("The training split has no nonconstant common-state features.")
    normalized_reference = normalize_common(reference, train_lower, train_upper).reshape(
        len(reference), -1
    )[:, nonconstant]
    normalized_generated = normalize_common(generated, train_lower, train_upper).reshape(
        len(generated), -1
    )[:, nonconstant]
    joint_wasserstein = joint_wasserstein1_subsampled(
        normalized_reference,
        normalized_generated,
        transport_samples,
        seed,
    )

    grid = physics_from_archive(archive)
    state = torch.from_numpy(generated.astype(np.float32))
    with torch.no_grad():
        signed_residual = ac_power_balance(state, grid).numpy()
        from_flow, to_flow = line_apparent_power(state, grid)
    absolute_residual = np.abs(signed_residual)
    sample_mean = absolute_residual.mean(axis=(1, 2))
    sample_max = absolute_residual.max(axis=(1, 2))
    sample_complex_mean = np.linalg.norm(signed_residual, axis=-1).mean(axis=1)

    lower = np.asarray(archive["common_lower"], dtype=np.float64)
    upper = np.asarray(archive["common_upper"], dtype=np.float64)
    bound_tolerance = np.asarray([1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6])
    bound_mask = (generated < lower[None, ...] - bound_tolerance) | (
        generated > upper[None, ...] + bound_tolerance
    )
    stored_range_violation = np.any(bound_mask, axis=(1, 2))
    # The source inequality set covers injections and voltage limits; phase
    # angle is retained only as a project range diagnostic.
    operational_bound_violation = np.any(bound_mask[..., :3], axis=(1, 2))
    bound_violation_per_channel = {
        name: float(np.any(bound_mask[..., channel], axis=1).mean())
        for channel, name in enumerate(CHANNELS)
    }

    limits = np.asarray(archive["branch_rate_pu"], dtype=np.float64)
    finite = np.isfinite(limits)
    thermal_violation_rate: float | None = None
    thermally_informative = bool(finite.any() and np.any(limits[finite] <= 10.0))
    if finite.any():
        thermal_violation = np.any(
            (from_flow.numpy()[:, finite] > limits[finite])
            | (to_flow.numpy()[:, finite] > limits[finite]),
            axis=1,
        )
        thermal_violation_rate = float(thermal_violation.mean())

    metadata = archive["metadata"]
    assert isinstance(metadata, dict)
    base_mva = float(metadata["base_mva"])
    delta_p_mw = signed_residual[..., 0] * base_mva
    delta_q_mvar = signed_residual[..., 1] * base_mva

    return {
        "dataset": str(data_path.resolve()),
        "generated": str(generated_path.resolve()),
        "method": sampling_metadata["method"],
        "reference_split": split_name,
        "reference_pool_count": int(len(pool)),
        "sample_count": int(len(generated)),
        "seed": int(seed),
        "metric_provenance": METRIC_PROVENANCE,
        "paper_metrics": {
            "joint_wasserstein1_normalized_subsampled": joint_wasserstein,
            "mean_complex_power_imbalance_pu": float(sample_complex_mean.mean()),
            "p95_sample_mean_complex_power_imbalance_pu": float(
                np.quantile(sample_complex_mean, 0.95)
            ),
            "per_bus_power_mismatch": {
                "delta_p_mw": {
                    "mean": delta_p_mw.mean(axis=0).tolist(),
                    "std": delta_p_mw.std(axis=0).tolist(),
                },
                "delta_q_mvar": {
                    "mean": delta_q_mvar.mean(axis=0).tolist(),
                    "std": delta_q_mvar.std(axis=0).tolist(),
                },
            },
        },
        "common_supplement_metrics": {
            "wasserstein_per_channel": marginal_wasserstein,
            "wasserstein_per_bus": marginal_wasserstein_per_bus,
            "absolute_mean_error_per_channel": mean_error,
            "generated_to_real_std_ratio": std_ratio,
            "mmd_rbf_squared": rbf_mmd_squared(
                reference, generated, metric_samples, seed
            ),
            "correlation_frobenius_error": correlation_error,
            "physics": {
                "mean_absolute_residual_pu": float(absolute_residual.mean()),
                "p95_absolute_residual_pu": float(
                    np.quantile(absolute_residual, 0.95)
                ),
                "p99_absolute_residual_pu": float(
                    np.quantile(absolute_residual, 0.99)
                ),
                "sample_mean_residual_p95_pu": float(
                    np.quantile(sample_mean, 0.95)
                ),
                "sample_max_residual_median_pu": float(np.median(sample_max)),
                "sample_max_residual_p95_pu": float(
                    np.quantile(sample_max, 0.95)
                ),
                "feasible_rate_by_tolerance": {
                    f"{value:g}": float(np.mean(sample_max <= value))
                    for value in (1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2)
                },
            },
            "constraint_violations": {
                "any_operational_bound_violation_rate": float(
                    operational_bound_violation.mean()
                ),
                "any_stored_range_violation_rate": float(
                    stored_range_violation.mean()
                ),
                "stored_bound_violation_rate_per_channel": bound_violation_per_channel,
                "absolute_numerical_tolerance_per_channel": bound_tolerance.tolist(),
                "voltage_violation_rate": bound_violation_per_channel["vm_pu"],
                "thermal_violation_rate": thermal_violation_rate,
                "thermal_limits_informative": thermally_informative,
            },
        },
        "sampling": sampling_metadata,
    }
