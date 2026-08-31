from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.metrics import (
    evaluate_generated,
    joint_wasserstein1_subsampled,
    normalize_common,
    reference_reference_baseline,
)


def test_normalize_common_maps_training_extrema() -> None:
    lower = np.asarray([[0.0, -2.0, 0.9, -0.1]])
    upper = np.asarray([[2.0, 2.0, 1.1, 0.1]])
    values = np.stack([lower, upper])
    normalized = normalize_common(values, lower, upper)
    assert np.allclose(normalized[0], -1.0)
    assert np.allclose(normalized[1], 1.0)


def test_joint_wasserstein_is_zero_for_identical_samples() -> None:
    values = np.arange(24, dtype=np.float64).reshape(3, 2, 4)
    report = joint_wasserstein1_subsampled(values, values, max_samples=3, seed=7)
    assert report["value"] == 0.0
    assert report["sample_count"] == 3


def _write_two_bus_archive(path: Path) -> None:
    state = np.zeros((6, 2, 4), dtype=np.float32)
    state[..., 2] = 1.0
    state[0, :, 2] = 0.99
    state[1, :, 2] = 1.01
    split = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    zero_matrix = np.zeros((2, 2), dtype=np.float64)
    np.savez_compressed(
        path,
        state_common=state,
        state_wang=np.zeros((6, 1), dtype=np.float32),
        split=split,
        common_lower=np.asarray([[-1.0, -1.0, 0.9, -1.0]] * 2, dtype=np.float32),
        common_upper=np.asarray([[1.0, 1.0, 1.1, 1.0]] * 2, dtype=np.float32),
        ybus_real=zero_matrix,
        ybus_imag=zero_matrix,
        yf_real=np.empty((0, 2), dtype=np.float64),
        yf_imag=np.empty((0, 2), dtype=np.float64),
        yt_real=np.empty((0, 2), dtype=np.float64),
        yt_imag=np.empty((0, 2), dtype=np.float64),
        branch_from=np.empty(0, dtype=np.int64),
        branch_to=np.empty(0, dtype=np.int64),
        branch_rate_pu=np.empty(0, dtype=np.float64),
        metadata_json=np.asarray(json.dumps({"base_mva": 100.0})),
    )


def _write_generated(path: Path, *, voltage: float = 1.0) -> None:
    generated = np.zeros((2, 2, 4), dtype=np.float32)
    generated[..., 2] = voltage
    metadata = {
        "method": "ddpm",
        "seconds_per_sample": 0.01,
        "network_function_evaluations_per_sample": 4,
        "physics_gradient_evaluations_per_sample": 0.0,
    }
    np.savez_compressed(
        path,
        generated_common=generated,
        generated_native=generated,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_evaluate_generated_reports_paper_metrics(tmp_path: Path) -> None:
    data = tmp_path / "data.npz"
    generated = tmp_path / "generated.npz"
    _write_two_bus_archive(data)
    _write_generated(generated)
    report = evaluate_generated(
        data,
        generated,
        split_name="test",
        metric_samples=2,
        transport_samples=2,
        seed=11,
    )
    assert report["paper_metrics"]["joint_wasserstein1_normalized_subsampled"]["value"] == 0.0
    assert report["paper_metrics"]["mean_complex_power_imbalance_pu"] == 0.0
    assert report["reference_split"] == "test"


def test_evaluate_generated_reports_exceedance_magnitude(tmp_path: Path) -> None:
    data = tmp_path / "data.npz"
    generated = tmp_path / "generated.npz"
    _write_two_bus_archive(data)
    _write_generated(generated, voltage=1.2)
    report = evaluate_generated(
        data,
        generated,
        split_name="test",
        metric_samples=2,
        transport_samples=2,
        seed=11,
    )
    constraints = report["common_supplement_metrics"]["constraint_violations"]
    voltage = constraints["stored_range_exceedance_by_channel"]["vm_pu"]
    assert voltage["sample_violation_rate"] == 1.0
    assert np.isclose(voltage["maximum_exceedance"], 0.1 - 1.0e-6)
    assert np.isclose(
        constraints["stored_pqv_sample_max_exceedance_p95"], 0.1 - 1.0e-6
    )


def _write_floor_archive(path: Path) -> None:
    rng = np.random.default_rng(17)
    state = rng.normal(size=(30, 2, 4)).astype(np.float32)
    state[..., 0:2] *= 0.05
    state[..., 2] = 1.0 + 0.01 * state[..., 2]
    state[..., 3] *= 0.02
    split = np.repeat(np.asarray([0, 1, 2], dtype=np.int8), 10)
    zero_matrix = np.zeros((2, 2), dtype=np.float64)
    np.savez_compressed(
        path,
        state_common=state,
        state_wang=np.zeros((30, 1), dtype=np.float32),
        split=split,
        common_lower=np.asarray([[-1.0, -1.0, 0.9, -1.0]] * 2, dtype=np.float32),
        common_upper=np.asarray([[1.0, 1.0, 1.1, 1.0]] * 2, dtype=np.float32),
        ybus_real=zero_matrix,
        ybus_imag=zero_matrix,
        yf_real=np.empty((0, 2), dtype=np.float64),
        yf_imag=np.empty((0, 2), dtype=np.float64),
        yt_real=np.empty((0, 2), dtype=np.float64),
        yt_imag=np.empty((0, 2), dtype=np.float64),
        branch_from=np.empty(0, dtype=np.int64),
        branch_to=np.empty(0, dtype=np.int64),
        branch_rate_pu=np.empty(0, dtype=np.float64),
        metadata_json=np.asarray(json.dumps({"base_mva": 100.0})),
    )


def test_reference_reference_baseline_is_repeated_and_finite(tmp_path: Path) -> None:
    data = tmp_path / "floor.npz"
    _write_floor_archive(data)
    floor = reference_reference_baseline(
        data,
        split_name="test",
        metric_samples=5,
        transport_samples=4,
        repeats=3,
        seed=9,
    )
    assert floor["subset_size_per_side"] == 5
    assert floor["repeats"] == 3
    for summary in floor["metrics"].values():
        assert np.isfinite(summary["mean"])
        assert summary["maximum"] >= summary["minimum"]
