"""Compare several generated power-flow datasets and create paper-ready figures.

Each ``--run`` points to either a run directory containing ``generated.npz``
and optionally ``training.json``, or directly to a generated NPZ file.  The
command recomputes all metrics under one data split, seed, and evaluator so
that different baselines are not compared using incompatible reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import load_archive
from evaluation.metrics import CHANNELS, SPLIT_CODES, evaluate_generated
from model.physics import ac_power_balance, physics_from_archive


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
LINESTYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v")
CHANNEL_LABELS = {
    "p_injection_pu": "Active injection (p.u.)",
    "q_injection_pu": "Reactive injection (p.u.)",
    "vm_pu": "Voltage magnitude (p.u.)",
    "theta_rad": "Voltage angle (rad)",
}


def _configure_style() -> None:
    """Use a compact colorblind-safe style that also works on headless servers."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _parse_run(specification: str) -> tuple[str, Path, Path | None]:
    if "=" not in specification:
        raise ValueError("Each --run must use LABEL=RUN_DIRECTORY_OR_GENERATED_NPZ.")
    label, raw_path = specification.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label:
        raise ValueError("Run label cannot be empty.")
    if path.is_dir():
        generated = path / "generated.npz"
        training = path / "training.json"
    else:
        generated = path
        training = path.parent / "training.json"
    if not generated.exists():
        raise FileNotFoundError(f"Missing generated samples for {label}: {generated}")
    return label, generated, training if training.exists() else None


def _safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "run"


def _load_generated(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        return source["generated_common"].astype(np.float64)


def _load_training(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(report: dict[str, Any], *keys: str) -> Any:
    value: Any = report
    for key in keys:
        value = value[key]
    return value


def _plot_value(value: Any, *, floor: float | None = None) -> float:
    """Convert optional JSON metrics to plot-safe floats."""

    if value is None:
        return float("nan")
    converted = float(value)
    if not np.isfinite(converted):
        return float("nan")
    return max(converted, floor) if floor is not None else converted


def _expanded_range(low: float, high: float) -> tuple[float, float]:
    """Give constant or nearly constant variables a non-singular plot range."""

    if high > low and not np.isclose(high, low):
        return low, high
    center = 0.5 * (low + high)
    padding = max(abs(center) * 0.01, 1.0e-6)
    return center - padding, center + padding


def _summary_row(
    label: str,
    report: dict[str, Any],
    training: dict[str, Any] | None,
) -> dict[str, Any]:
    constraints = _nested(
        report, "common_supplement_metrics", "constraint_violations"
    )
    sampling = report["sampling"]
    return {
        "label": label,
        "method": report["method"],
        "sample_count": report["sample_count"],
        "seed": report["seed"],
        "parameter_count": None if training is None else training.get("parameter_count"),
        "best_validation_total": (
            None if training is None else training.get("best_validation_total")
        ),
        "training_seconds": None if training is None else training.get("training_seconds"),
        "joint_wasserstein1": _nested(
            report,
            "paper_metrics",
            "joint_wasserstein1_normalized_subsampled",
            "value",
        ),
        "wang_mean_complex_imbalance_pu": _nested(
            report, "paper_metrics", "mean_complex_power_imbalance_pu"
        ),
        "mmd_rbf_squared": _nested(
            report, "common_supplement_metrics", "mmd_rbf_squared"
        ),
        "correlation_frobenius_error": _nested(
            report, "common_supplement_metrics", "correlation_frobenius_error"
        ),
        "sample_max_residual_p95_pu": _nested(
            report,
            "common_supplement_metrics",
            "physics",
            "sample_max_residual_p95_pu",
        ),
        "operational_bound_violation_rate": constraints[
            "any_operational_bound_violation_rate"
        ],
        "stored_range_violation_rate": constraints[
            "any_stored_range_violation_rate"
        ],
        "voltage_violation_rate": constraints["voltage_violation_rate"],
        "thermal_violation_rate": (
            constraints["thermal_violation_rate"]
            if constraints["thermal_limits_informative"]
            else None
        ),
        "thermal_limits_informative": constraints["thermal_limits_informative"],
        "seconds_per_sample": sampling.get("seconds_per_sample"),
        "network_function_evaluations_per_sample": sampling.get(
            "network_function_evaluations_per_sample"
        ),
        "physics_gradient_evaluations_per_sample": sampling.get(
            "physics_gradient_evaluations_per_sample"
        ),
    }


def _write_tables(
    output_dir: Path,
    rows: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "interpretation": (
            "All reported quality metrics are lower-is-better except feasibility rates. "
            "No weighted overall score is computed because statistical fidelity, physics, "
            "and speed have application-dependent trade-offs."
        ),
        "single_seed_warning": "Uncertainty bars require the planned multi-seed runs.",
        "runs": rows,
        "metric_provenance": next(iter(reports.values()))["metric_provenance"],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value: Any, digits: int = 5) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, int):
            return f"{value:,}"
        return f"{float(value):.{digits}g}"

    lines = [
        "# Baseline comparison",
        "",
        "All quality and cost columns are lower-is-better. Results currently use one seed; "
        "therefore no uncertainty interval or significance claim is reported.",
        "",
        "| Run | Params | Joint W1 | MMD² | Mean imbalance (p.u.) | Max residual P95 (p.u.) | Bound violation | Sampling (ms/sample) | NFE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        milliseconds = (
            None
            if row["seconds_per_sample"] is None
            else 1_000.0 * float(row["seconds_per_sample"])
        )
        lines.append(
            "| {label} | {params} | {w1} | {mmd} | {imbalance} | {p95} | {violation} | {ms} | {nfe} |".format(
                label=row["label"],
                params=fmt(row["parameter_count"], 0),
                w1=fmt(row["joint_wasserstein1"]),
                mmd=fmt(row["mmd_rbf_squared"]),
                imbalance=fmt(row["wang_mean_complex_imbalance_pu"]),
                p95=fmt(row["sample_max_residual_p95_pu"]),
                violation=fmt(row["operational_bound_violation_rate"]),
                ms=fmt(milliseconds),
                nfe=fmt(row["network_function_evaluations_per_sample"], 0),
            )
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- [Overview](figures/overview.png)",
            "- [Marginal distributions](figures/marginal_distributions.png)",
            "- [Joint distributions](figures/joint_distributions.png)",
            "- [Per-bus power mismatch](figures/power_mismatch_by_bus.png)",
            "- [Residual CDF](figures/residual_cdf.png)",
            "- [Training convergence](figures/training_convergence.png)",
            "",
            "The joint W1 metric follows Hoseinpour--Dvorkin Eq. (33) on a seeded "
            "subsample. Mean complex imbalance follows Wang et al. Eq. (7). MMD, "
            "correlation error, residual quantiles, and speed are common project supplements.",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _grouped_bars(
    ax: plt.Axes,
    labels: list[str],
    series: list[tuple[str, list[float]]],
    *,
    ylabel: str,
    log: bool = False,
) -> None:
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.8 / max(len(series), 1)
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=name, color=COLORS[index])
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(frameon=False)


def _plot_overview(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["label"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    _grouped_bars(
        axes[0, 0],
        labels,
        [
            (
                "Joint W1",
                [_plot_value(row["joint_wasserstein1"], floor=1.0e-12) for row in rows],
            ),
            (
                "MMD²",
                [_plot_value(row["mmd_rbf_squared"], floor=1.0e-12) for row in rows],
            ),
            (
                "Correlation error",
                [
                    _plot_value(row["correlation_frobenius_error"], floor=1.0e-12)
                    for row in rows
                ],
            ),
        ],
        ylabel="Statistical distance",
        log=True,
    )
    axes[0, 0].set_title("A  Statistical fidelity (lower is better)", loc="left")

    _grouped_bars(
        axes[0, 1],
        labels,
        [
            (
                "Mean complex imbalance",
                [
                    _plot_value(row["wang_mean_complex_imbalance_pu"], floor=1.0e-12)
                    for row in rows
                ],
            ),
            (
                "Max residual P95",
                [
                    _plot_value(row["sample_max_residual_p95_pu"], floor=1.0e-12)
                    for row in rows
                ],
            ),
        ],
        ylabel="Power mismatch (p.u.)",
        log=True,
    )
    axes[0, 1].set_title("B  Physical consistency (lower is better)", loc="left")

    _grouped_bars(
        axes[1, 0],
        labels,
        [
            (
                "P/Q/V operational bounds",
                [
                    100.0 * float(row["operational_bound_violation_rate"])
                    for row in rows
                ],
            ),
            (
                "Voltage",
                [100.0 * float(row["voltage_violation_rate"]) for row in rows],
            ),
        ],
        ylabel="Violating samples (%)",
    )
    axes[1, 0].set_title("C  Inequality constraints (lower is better)", loc="left")

    ax = axes[1, 1]
    for index, row in enumerate(rows):
        seconds = max(float(row["seconds_per_sample"] or 0.0), 1.0e-9)
        imbalance = max(float(row["wang_mean_complex_imbalance_pu"]), 1.0e-12)
        parameter_count = float(row["parameter_count"] or 1.0)
        size = 35.0 + 80.0 * math.sqrt(parameter_count / max(float(r["parameter_count"] or 1.0) for r in rows))
        ax.scatter(
            1_000.0 * seconds,
            imbalance,
            s=size,
            color=COLORS[index],
            marker=MARKERS[index],
            label=str(row["label"]),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sampling time (ms/sample)")
    ax.set_ylabel("Mean complex imbalance (p.u.)")
    ax.set_title("D  Speed--physics trade-off", loc="left")
    ax.legend(frameon=False)
    fig.suptitle("Single-seed baseline dashboard; marker area reflects parameter count", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "overview")


def _reference_pool(archive: dict[str, object], split_name: str) -> np.ndarray:
    split = np.asarray(archive["split"], dtype=np.int8)
    return np.asarray(archive["state_common"], dtype=np.float64)[
        split == SPLIT_CODES[split_name]
    ]


def _plot_marginals(
    output_dir: Path,
    reference: np.ndarray,
    generated: dict[str, np.ndarray],
    plot_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    reference = reference[rng.choice(len(reference), min(plot_samples, len(reference)), replace=False)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for channel, (name, ax) in enumerate(zip(CHANNELS, axes.flat, strict=True)):
        bus = int(np.argmax(reference[..., channel].var(axis=0)))
        arrays = [("Real", reference[:, bus, channel], "#000000")]
        for index, (label, values) in enumerate(generated.items()):
            count = min(plot_samples, len(values))
            indices = rng.choice(len(values), count, replace=False)
            arrays.append((label, values[indices, bus, channel], COLORS[index]))
        all_values = np.concatenate([values for _, values, _ in arrays])
        low, high = np.quantile(all_values, [0.005, 0.995])
        low, high = _expanded_range(float(low), float(high))
        bins = np.linspace(low, high, 45)
        for index, (label, values, color) in enumerate(arrays):
            plot_values = np.clip(values, low, high)
            ax.hist(
                plot_values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.2,
                linestyle=LINESTYLES[index % len(LINESTYLES)],
                color=color,
                label=label,
            )
        ax.set_xlabel(CHANNEL_LABELS[name])
        ax.set_ylabel("Density")
        ax.set_title(f"{chr(65 + channel)}  Bus {bus + 1}", loc="left")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Marginal distributions at the highest-variance bus for each channel", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "marginal_distributions")


def _plot_joint_distributions(
    output_dir: Path,
    reference: np.ndarray,
    generated: dict[str, np.ndarray],
    plot_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    count = min(plot_samples, len(reference))
    real = reference[rng.choice(len(reference), count, replace=False)]
    p_q_bus = int(np.argmax(real[..., :2].var(axis=0).sum(axis=1)))
    v_theta_bus = int(np.argmax(real[..., 2:].var(axis=0).sum(axis=1)))
    columns = [("Real", real)] + list(generated.items())
    fig, axes = plt.subplots(2, len(columns), figsize=(2.25 * len(columns), 4.6), squeeze=False)

    def limits(channel_x: int, channel_y: int, bus: int) -> tuple[float, float, float, float]:
        values = [real[:, bus, [channel_x, channel_y]]]
        for array in generated.values():
            take = array[rng.choice(len(array), min(count, len(array)), replace=False)]
            values.append(take[:, bus, [channel_x, channel_y]])
        joined = np.concatenate(values, axis=0)
        x_low, y_low = np.quantile(joined, 0.005, axis=0)
        x_high, y_high = np.quantile(joined, 0.995, axis=0)
        x_low, x_high = _expanded_range(float(x_low), float(x_high))
        y_low, y_high = _expanded_range(float(y_low), float(y_high))
        return x_low, x_high, y_low, y_high

    pair_specs = [
        (0, 1, p_q_bus, "P injection (p.u.)", "Q injection (p.u.)"),
        (2, 3, v_theta_bus, "Voltage magnitude (p.u.)", "Voltage angle (rad)"),
    ]
    for row, (x_channel, y_channel, bus, x_label, y_label) in enumerate(pair_specs):
        x_low, x_high, y_low, y_high = limits(x_channel, y_channel, bus)
        for column, (label, values) in enumerate(columns):
            take = values[rng.choice(len(values), min(count, len(values)), replace=False)]
            ax = axes[row, column]
            ax.hexbin(
                take[:, bus, x_channel],
                take[:, bus, y_channel],
                gridsize=30,
                extent=(x_low, x_high, y_low, y_high),
                mincnt=1,
                cmap="cividis",
            )
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)
            ax.set_xlabel(x_label)
            if column == 0:
                ax.set_ylabel(y_label)
            ax.set_title(f"{label}\nBus {bus + 1}")
    fig.suptitle("Joint distributions and density", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "joint_distributions")


def _plot_power_mismatch(
    output_dir: Path,
    reports: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    specs = [
        ("delta_p_mw", "Active mismatch (MW)"),
        ("delta_q_mvar", "Reactive mismatch (MVar)"),
    ]
    for axis_index, (key, ylabel) in enumerate(specs):
        ax = axes[axis_index]
        for index, (label, report) in enumerate(reports.items()):
            mismatch = _nested(
                report, "paper_metrics", "per_bus_power_mismatch", key
            )
            mean = np.asarray(mismatch["mean"], dtype=np.float64)
            std = np.asarray(mismatch["std"], dtype=np.float64)
            buses = np.arange(1, len(mean) + 1)
            ax.plot(
                buses,
                mean,
                color=COLORS[index],
                linestyle=LINESTYLES[index],
                marker=MARKERS[index],
                markersize=3,
                label=label,
            )
            ax.fill_between(buses, mean - std, mean + std, color=COLORS[index], alpha=0.12)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{chr(65 + axis_index)}  Mean ± standard deviation", loc="left")
    axes[0].legend(frameon=False, ncol=max(1, len(reports)))
    axes[1].set_xlabel("Bus index")
    fig.suptitle("Per-bus signed power mismatches (Hoseinpour--Dvorkin protocol)", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "power_mismatch_by_bus")


def _plot_residual_cdf(
    output_dir: Path,
    archive: dict[str, object],
    generated: dict[str, np.ndarray],
) -> None:
    grid = physics_from_archive(archive)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for index, (label, values) in enumerate(generated.items()):
        with torch.no_grad():
            residual = ac_power_balance(
                torch.from_numpy(values.astype(np.float32)), grid
            ).numpy()
        complex_mean = np.linalg.norm(residual, axis=-1).mean(axis=1)
        max_component = np.abs(residual).max(axis=(1, 2))
        for ax, samples in zip(axes, (complex_mean, max_component), strict=True):
            samples = np.sort(np.maximum(samples, 1.0e-12))
            probability = np.arange(1, len(samples) + 1) / len(samples)
            ax.plot(
                samples,
                probability,
                color=COLORS[index],
                linestyle=LINESTYLES[index],
                marker=MARKERS[index],
                markevery=max(1, len(samples) // 20),
                markersize=3,
                label=label,
            )
    axes[0].set_xlabel("Sample mean complex imbalance (p.u.)")
    axes[1].set_xlabel("Sample maximum component mismatch (p.u.)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylabel("Empirical CDF")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_title("A  Wang Eq. (7) distribution", loc="left")
    axes[1].set_title("B  Worst-bus residual distribution", loc="left")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "residual_cdf")


def _plot_training(
    output_dir: Path,
    training_records: dict[str, dict[str, Any] | None],
) -> None:
    available = {label: record for label, record in training_records.items() if record}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    if not available:
        for ax in axes:
            ax.text(0.5, 0.5, "training.json not available", ha="center", va="center")
            ax.set_axis_off()
    else:
        for index, (label, record) in enumerate(available.items()):
            assert record is not None
            history = record.get("history", [])
            epochs = np.asarray([item["epoch"] for item in history])
            train = np.asarray([item["train"]["total"] for item in history], dtype=np.float64)
            validation = np.asarray(
                [item["validation"]["total"] for item in history], dtype=np.float64
            )
            axes[0].plot(
                epochs,
                train / max(train[0], 1.0e-12),
                color=COLORS[index],
                linestyle=LINESTYLES[index],
                marker=MARKERS[index],
                markevery=max(1, len(epochs) // 10),
                markersize=3,
                label=label,
            )
            axes[1].plot(
                epochs,
                validation / max(validation[0], 1.0e-12),
                color=COLORS[index],
                linestyle=LINESTYLES[index],
                marker=MARKERS[index],
                markevery=max(1, len(epochs) // 10),
                markersize=3,
                label=label,
            )
        axes[0].set_title("A  Relative training objective", loc="left")
        axes[1].set_title("B  Relative validation objective", loc="left")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss / epoch-1 loss")
            ax.legend(frameon=False)
    fig.suptitle("Convergence only; absolute objectives differ across methods", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "figures" / "training_convergence")


def compare(args: argparse.Namespace) -> None:
    _configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed = [_parse_run(item) for item in args.run]
    labels = [item[0] for item in parsed]
    if len(set(labels)) != len(labels):
        raise ValueError("Every --run label must be unique.")

    reports: dict[str, dict[str, Any]] = {}
    generated: dict[str, np.ndarray] = {}
    training_records: dict[str, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for label, generated_path, training_path in parsed:
        report = evaluate_generated(
            args.data,
            generated_path,
            split_name=args.split,
            metric_samples=args.metric_samples,
            transport_samples=args.transport_samples,
            seed=args.seed,
        )
        reports[label] = report
        generated[label] = _load_generated(generated_path)
        training = _load_training(training_path)
        training_records[label] = training
        rows.append(_summary_row(label, report, training))
        (metrics_dir / f"{_safe_name(label)}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    _write_tables(args.output_dir, rows, reports)
    archive = load_archive(args.data)
    reference = _reference_pool(archive, args.split)
    _plot_overview(args.output_dir, rows)
    _plot_marginals(args.output_dir, reference, generated, args.plot_samples, args.seed)
    _plot_joint_distributions(
        args.output_dir, reference, generated, args.plot_samples, args.seed
    )
    _plot_power_mismatch(args.output_dir, reports)
    _plot_residual_cdf(args.output_dir, archive, generated)
    _plot_training(args.output_dir, training_records)
    print(f"Comparison report: {(args.output_dir / 'comparison.md').resolve()}")
    print(f"Figures: {(args.output_dir / 'figures').resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeat as LABEL=RUN_DIRECTORY_OR_GENERATED_NPZ.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--metric-samples", type=int, default=1_000)
    parser.add_argument("--transport-samples", type=int, default=256)
    parser.add_argument("--plot-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()
