"""Train ordinary DDPM, Wang PI-DDPM, or Hoseinpour constrained diffusion."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import PowerFlowDataset, load_archive
from model.factory import create_model
from model.physics import mean_complex_imbalance, physics_from_archive, wang_to_common
from model.scaling import TensorMinMaxScaler
from model.wang import WangScheduleNetwork, estimate_terminal_imbalance


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _fit_scaler(
    archive: dict[str, object], method: str
) -> tuple[TensorMinMaxScaler, str]:
    representation = "wang" if method == "wang" else "common"
    key = "state_wang" if method == "wang" else "state_common"
    states = torch.from_numpy(np.asarray(archive[key], dtype=np.float32))
    split = np.asarray(archive["split"], dtype=np.int8)
    train = states[torch.from_numpy(split == 0)]
    if method == "wang":
        return TensorMinMaxScaler.fit(train, 0.0, 1.0), representation
    return TensorMinMaxScaler.fit(train, -1.0, 1.0), representation


def _learn_wang_schedule(
    archive: dict[str, object],
    scaler: TensorMinMaxScaler,
    loader: DataLoader[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    gamma_terminal: float,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """Fit the paper's auxiliary schedule before the denoiser."""

    input_dim = np.asarray(archive["state_wang"]).shape[1]
    learner = WangScheduleNetwork(
        input_dim, args.steps, args.beta_start, args.beta_end
    ).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.schedule_learning_rate)
    grid = physics_from_archive(archive).to(device)
    load_buses = torch.from_numpy(
        np.asarray(archive["load_bus_indices"], dtype=np.int64)
    ).to(device)
    generator_buses = torch.from_numpy(
        np.asarray(archive["generator_bus_indices"], dtype=np.int64)
    ).to(device)
    history = {"schedule_loss": []}
    for epoch in range(args.schedule_epochs):
        losses: list[float] = []
        for native in loader:
            clean = scaler.transform(native.to(device))
            alpha_bars = learner(clean)
            # Evaluating all T states is expensive. Sampling several t values
            # is an unbiased stochastic version of Eq. (11).
            count = min(args.schedule_time_samples, args.steps)
            timestep = torch.randint(args.steps, (count,), device=device)
            clean_subset = clean[: min(len(clean), count)]
            timestep = timestep[: len(clean_subset)]
            noise = torch.randn_like(clean_subset)
            alpha = alpha_bars[timestep].view(-1, 1)
            noisy = alpha.sqrt() * clean_subset + (1.0 - alpha).sqrt() * noise
            native_noisy = scaler.inverse(noisy)
            common = wang_to_common(
                native_noisy,
                load_buses,
                generator_buses,
                grid.ybus.shape[0],
                grid.base_mva,
            )
            actual = mean_complex_imbalance(common, grid)
            target = gamma_terminal * (timestep.float() + 1.0) / args.steps
            loss = (actual - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history["schedule_loss"].append(float(np.mean(losses)))
        print(f"schedule epoch={epoch + 1} loss={history['schedule_loss'][-1]:.6f}")

    # Paper averages network outputs across the dataset. A deterministic train
    # batch is used here and its size is recorded in the checkpoint.
    reference_native = next(iter(loader)).to(device)
    with torch.no_grad():
        alpha_bars = learner(scaler.transform(reference_native)).detach().cpu()
    return alpha_bars, history


def _losses(model: torch.nn.Module, method: str, normalized: torch.Tensor) -> dict[str, torch.Tensor]:
    if method == "ddpm":
        loss = model.noise_loss(normalized)  # type: ignore[attr-defined]
        return {"total": loss, "noise": loss}
    return model.training_losses(normalized)  # type: ignore[attr-defined]


def run_epoch(
    model: torch.nn.Module,
    method: str,
    loader: DataLoader[torch.Tensor],
    scaler: TensorMinMaxScaler,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    sample_count = 0
    for native in loader:
        normalized = scaler.transform(native.to(device))
        with torch.set_grad_enabled(training):
            losses = _losses(model, method, normalized)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        batch = len(native)
        sample_count += batch
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * batch
    return {name: value / sample_count for name, value in sums.items()}


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    archive = load_archive(args.data)
    scaler, representation = _fit_scaler(archive, args.method)
    train_set = PowerFlowDataset(archive, representation, "train")
    validation_set = PowerFlowDataset(archive, representation, "validation")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    grid = physics_from_archive(archive).to(device)
    alpha_bars = None
    schedule_history: dict[str, list[float]] = {}
    gamma_terminal = args.gamma_terminal
    if args.method == "wang":
        if gamma_terminal <= 0.0:
            gamma_terminal = estimate_terminal_imbalance(
                scaler,
                grid,
                torch.from_numpy(np.asarray(archive["load_bus_indices"], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(archive["generator_bus_indices"], dtype=np.int64)).to(device),
                np.asarray(archive["state_wang"]).shape[1],
            )
        if args.schedule_epochs > 0:
            alpha_bars, schedule_history = _learn_wang_schedule(
                archive, scaler, train_loader, args, device, gamma_terminal
            )

    model_config: dict[str, object] = {
        "method": args.method,
        "steps": args.steps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "hidden": args.hidden,
        "layers": args.layers,
        "physics_weight": args.physics_weight,
        "gamma_terminal": gamma_terminal,
        "wang_sigmoid_output": args.wang_sigmoid_output,
        "representation": representation,
        "provenance": {
            "ddpm_architecture": "project-controlled baseline",
            "wang_architecture": "paper-stated widths; attention implementation inferred",
            "hoseinpour_architecture": "inferred because author code and architecture are absent",
        },
    }
    model = create_model(archive, scaler, model_config, device, alpha_bars)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, args.method, train_loader, scaler, device, optimizer
        )
        validation_metrics = run_epoch(
            model, args.method, validation_loader, scaler, device, None
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch} train={train_metrics['total']:.6f} "
            f"validation={validation_metrics['total']:.6f}"
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "scaler": scaler.state_dict(),
            "alpha_bars": alpha_bars,
            "epoch": epoch,
            "seed": args.seed,
            "data_path": str(args.data.resolve()),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            torch.save(checkpoint, args.output_dir / "best.pt")

    summary = {
        "method": args.method,
        "best_validation_total": best,
        "training_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "model_config": model_config,
        "history": history,
        "schedule_history": schedule_history,
    }
    (args.output_dir / "training.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ddpm", "wang", "hoseinpour"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1.0e-4)
    parser.add_argument("--beta-end", type=float, default=2.0e-2)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--gamma-terminal", type=float, default=-1.0)
    parser.add_argument("--schedule-epochs", type=int, default=20)
    parser.add_argument("--schedule-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--schedule-time-samples", type=int, default=16)
    parser.add_argument("--wang-sigmoid-output", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
