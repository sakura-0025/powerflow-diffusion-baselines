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


def sample(args: argparse.Namespace) -> None:
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
            physical, evaluations = model.sample(  # type: ignore[attr-defined]
                batch,
                device,
                guidance_scale=args.guidance_scale,
                inequality_weight=args.inequality_weight,
            )
            native = physical
            # The model returns the number of guidance calls along one sample
            # trajectory; total PGE also includes the batch cardinality.
            physics_evaluations += evaluations * batch
        common_blocks.append(physical.detach().cpu().numpy().astype(np.float32))
        native_blocks.append(native.detach().cpu().numpy().astype(np.float32))
        remaining -= batch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated_common = np.concatenate(common_blocks)
    generated_native = np.concatenate(native_blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        "guidance_scale": args.guidance_scale if method == "hoseinpour" else 0.0,
    }
    np.savez_compressed(
        args.output,
        generated_common=generated_common,
        generated_native=generated_native,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=5.0e-4)
    parser.add_argument("--inequality-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    sample(parse_args())


if __name__ == "__main__":
    main()
