"""Construct reproducible model instances from checkpoint metadata."""

from __future__ import annotations

import numpy as np
import torch

from .ddpm import DiffusionConfig, GaussianDiffusion, VectorDenoiser
from .hoseinpour import HoseinpourConstrainedDiffusion
from .physics import physics_from_archive
from .scaling import TensorMinMaxScaler
from .wang import WangPhysicsInformedDDPM


def create_model(
    archive: dict[str, object],
    scaler: TensorMinMaxScaler,
    config: dict[str, object],
    device: torch.device,
    alpha_bars: torch.Tensor | None = None,
) -> torch.nn.Module:
    method = str(config["method"])
    diffusion = DiffusionConfig(
        steps=int(config["steps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )
    grid = physics_from_archive(archive).to(device)
    bus_count = grid.ybus.shape[0]
    if method == "ddpm":
        input_dim = int(np.prod(np.asarray(archive["state_common"]).shape[1:]))
        model: torch.nn.Module = GaussianDiffusion(
            VectorDenoiser(
                input_dim,
                hidden=int(config["hidden"]),
                layers=int(config["layers"]),
            ),
            diffusion,
        )
    elif method == "wang":
        input_dim = np.asarray(archive["state_wang"]).shape[1]
        model = WangPhysicsInformedDDPM(
            input_dim,
            diffusion,
            scaler,
            grid,
            torch.from_numpy(np.asarray(archive["load_bus_indices"], dtype=np.int64)).to(device),
            torch.from_numpy(np.asarray(archive["generator_bus_indices"], dtype=np.int64)).to(device),
            alpha_bars=alpha_bars,
            physics_weight=float(config["physics_weight"]),
            gamma_terminal=float(config["gamma_terminal"]),
            sigmoid_output=bool(config["wang_sigmoid_output"]),
        )
    elif method == "hoseinpour":
        model = HoseinpourConstrainedDiffusion(
            bus_count,
            diffusion,
            scaler,
            grid,
            hidden=int(config["hidden"]),
            layers=int(config["layers"]),
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return model.to(device)

