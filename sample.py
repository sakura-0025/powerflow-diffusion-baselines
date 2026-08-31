"""Generate samples from a trained baseline checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data.dataset import load_archive
from model.factory import create_model
from model.scaling import TensorMinMaxScaler


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _merge_guidance_trace(
    accumulator: dict[int, dict[str, float]],
    trace: list[dict[str, float | int]],
) -> None:
    """Aggregate per-batch guidance diagnostics without retaining huge logs."""

    mean_fields = (
        "residual_before_mean",
        "residual_after_mean",
        "residual_reduction_fraction_mean",
        "harmful_update_fraction",
        "raw_gradient_norm_mean",
        "applied_gradient_norm_mean",
        "relative_update_norm_mean",
    )
    for item in trace:
        step = int(item["step"])
        active = int(item["active_samples"])
        target = accumulator.setdefault(
            step,
            {
                "alpha_bar": float(item["alpha_bar"]),
                "active_samples": 0.0,
                "total_samples": 0.0,
                **{f"{field}_weighted": 0.0 for field in mean_fields},
            },
        )
        target["active_samples"] += active
        target["total_samples"] += int(item["total_samples"])
        for field in mean_fields:
            target[f"{field}_weighted"] += float(item[field]) * active


def _finalize_guidance_trace(
    accumulator: dict[int, dict[str, float]],
) -> list[dict[str, float | int]]:
    """Convert weighted accumulators to a compact one-row-per-step trace."""

    mean_fields = (
        "residual_before_mean",
        "residual_after_mean",
        "residual_reduction_fraction_mean",
        "harmful_update_fraction",
        "raw_gradient_norm_mean",
        "applied_gradient_norm_mean",
        "relative_update_norm_mean",
    )
    result: list[dict[str, float | int]] = []
    for step in sorted(accumulator, reverse=True):
        item = accumulator[step]
        active = int(item["active_samples"])
        total = int(item["total_samples"])
        row: dict[str, float | int] = {
            "step": step,
            "alpha_bar": item["alpha_bar"],
            "active_samples": active,
            "total_samples": total,
            "active_fraction": active / max(total, 1),
        }
        for field in mean_fields:
            row[field] = item[f"{field}_weighted"] / max(active, 1)
        result.append(row)
    return result


def sample(args: argparse.Namespace) -> None:
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("num-samples and batch-size must be positive.")
    if args.guidance_scale < 0.0 or args.inequality_weight < 0.0:
        raise ValueError("guidance-scale and inequality-weight cannot be negative.")
    if args.guidance_last_steps is not None and args.guidance_last_steps < 1:
        raise ValueError("guidance-last-steps must be positive when provided.")
    if not 0.0 <= args.guidance_alpha_bar_min <= 1.0:
        raise ValueError("guidance-alpha-bar-min must lie in [0, 1].")
    if args.guidance_residual_threshold < 0.0:
        raise ValueError("guidance-residual-threshold cannot be negative.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    archive = load_archive(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scaler = TensorMinMaxScaler.from_state_dict(checkpoint["scaler"])
    config = checkpoint["model_config"]
    method = str(config["method"])
    model = create_model(
        archive,
        scaler,
        config,
        device,
        checkpoint.get("alpha_bars"),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    common_blocks: list[np.ndarray] = []
    native_blocks: list[np.ndarray] = []
    physics_evaluations = 0
    physics_gradient_batch_calls = 0
    guidance_trace_accumulator: dict[int, dict[str, float]] = {}
    started = time.perf_counter()
    remaining = args.num_samples
    while remaining:
        batch = min(remaining, args.batch_size)
        if method == "ddpm":
            bus_count = np.asarray(archive["state_common"]).shape[1]
            normalized = model.sample((batch, bus_count, 4), device)  # type: ignore[attr-defined]
            physical = scaler.inverse(normalized)
            native = physical
        elif method == "wang":
            input_dim = np.asarray(archive["state_wang"]).shape[1]
            normalized = model.diffusion.sample((batch, input_dim), device)  # type: ignore[attr-defined]
            native = scaler.inverse(normalized)
            physical = model.decode_common(normalized)  # type: ignore[attr-defined]
        else:
            physical, diagnostics = model.sample(  # type: ignore[attr-defined]
                batch,
                device,
                guidance_scale=args.guidance_scale,
                inequality_weight=args.inequality_weight,
                guidance_last_steps=args.guidance_last_steps,
                guidance_alpha_bar_min=args.guidance_alpha_bar_min,
                guidance_residual_threshold=args.guidance_residual_threshold,
                normalize_guidance_gradient=args.normalize_guidance_gradient,
            )
            native = physical
            physics_evaluations += int(
                diagnostics["physics_gradient_evaluations_total"]
            )
            physics_gradient_batch_calls += int(
                diagnostics["physics_gradient_batch_calls"]
            )
            _merge_guidance_trace(
                guidance_trace_accumulator,
                diagnostics["trace"],  # type: ignore[arg-type]
            )
        common_blocks.append(physical.detach().cpu().numpy().astype(np.float32))
        native_blocks.append(native.detach().cpu().numpy().astype(np.float32))
        remaining -= batch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated_common = np.concatenate(common_blocks)
    generated_native = np.concatenate(native_blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    guidance_extension_enabled = method == "hoseinpour" and (
        args.guidance_last_steps is not None
        or args.guidance_alpha_bar_min > 0.0
        or args.guidance_residual_threshold > 0.0
        or args.normalize_guidance_gradient
    )
    guidance_trace = _finalize_guidance_trace(guidance_trace_accumulator)
    metadata = {
        "method": method,
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "sampling_seconds": elapsed,
        "seconds_per_sample": elapsed / args.num_samples,
        "network_function_evaluations_per_sample": (
            int(config["steps"]) * (2 if method == "hoseinpour" else 1)
        ),
        "physics_gradient_evaluations_total": physics_evaluations,
        "physics_gradient_evaluations_per_sample": physics_evaluations
        / args.num_samples,
        "physics_gradient_batch_calls_total": physics_gradient_batch_calls,
        "guidance_scale": args.guidance_scale if method == "hoseinpour" else 0.0,
        "guidance_protocol": (
            "project_sparse_diagnostic_extension"
            if guidance_extension_enabled
            else (
                "paper_aligned_fixed_all_steps"
                if args.guidance_scale > 0.0
                else "unguided_ablation"
            )
        )
        if method == "hoseinpour"
        else "not_applicable",
        "guidance_last_steps": args.guidance_last_steps,
        "guidance_alpha_bar_min": args.guidance_alpha_bar_min,
        "guidance_residual_threshold": args.guidance_residual_threshold,
        "normalize_guidance_gradient": args.normalize_guidance_gradient,
        "guidance_trace": guidance_trace,
    }
    np.savez_compressed(
        args.output,
        generated_common=generated_common,
        generated_native=generated_native,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    if args.diagnostics_output is not None:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.write_text(
            json.dumps(
                {
                    "provenance": (
                        "Project-controlled diagnostic extension; not reported in "
                        "Hoseinpour--Dvorkin."
                    ),
                    "sampling": metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    printable = dict(metadata)
    printable["guidance_trace_steps"] = len(guidance_trace)
    printable.pop("guidance_trace", None)
    print(json.dumps(printable, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=5.0e-4)
    parser.add_argument("--inequality-weight", type=float, default=1.0)
    parser.add_argument(
        "--guidance-last-steps",
        type=int,
        help=(
            "Project extension: apply physics gradients only to the final K reverse steps."
        ),
    )
    parser.add_argument(
        "--guidance-alpha-bar-min",
        type=float,
        default=0.0,
        help="Project extension: guide only where alpha_bar is at least this value.",
    )
    parser.add_argument(
        "--guidance-residual-threshold",
        type=float,
        default=0.0,
        help="Project extension: skip samples below this constraint-energy threshold.",
    )
    parser.add_argument(
        "--normalize-guidance-gradient",
        action="store_true",
        help="Project extension: normalize each sample's joint guidance gradient.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Optional JSON copy of the aggregated timestep guidance trace.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    sample(parse_args())


if __name__ == "__main__":
    main()
