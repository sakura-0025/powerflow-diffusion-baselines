"""Hoseinpour-Dvorkin constrained diffusion with variable decoupling.

This follows Algorithms 4-5: two denoisers model (p,theta) and (q,v), all
features are min-max normalized to [-1,1], and equality/inequality gradients
modify each predicted clean sample before the DDPM posterior step.
"""

from __future__ import annotations

import torch
from torch import nn

from .ddpm import DiffusionConfig, GaussianDiffusion, VectorDenoiser
from .physics import GridPhysics, equality_loss, inequality_loss
from .scaling import TensorMinMaxScaler


class HoseinpourConstrainedDiffusion(nn.Module):
    def __init__(
        self,
        bus_count: int,
        diffusion_config: DiffusionConfig,
        scaler: TensorMinMaxScaler,
        grid: GridPhysics,
        hidden: int = 512,
        layers: int = 6,
    ) -> None:
        super().__init__()
        self.bus_count = bus_count
        self.scaler = scaler
        self.grid = grid
        self.p_theta = GaussianDiffusion(
            VectorDenoiser(2 * bus_count, hidden=hidden, layers=layers), diffusion_config
        )
        self.q_v = GaussianDiffusion(
            VectorDenoiser(2 * bus_count, hidden=hidden, layers=layers), diffusion_config
        )

    def split(self, common: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.cat([common[..., 0], common[..., 3]], dim=-1),
            torch.cat([common[..., 1], common[..., 2]], dim=-1),
        )

    def combine(self, p_theta: torch.Tensor, q_v: torch.Tensor) -> torch.Tensor:
        n = self.bus_count
        return torch.stack(
            [p_theta[:, :n], q_v[:, :n], q_v[:, n:], p_theta[:, n:]], dim=-1
        )

    def training_losses(self, normalized_clean: torch.Tensor) -> dict[str, torch.Tensor]:
        first, second = self.split(normalized_clean)
        first_loss = self.p_theta.noise_loss(first)
        second_loss = self.q_v.noise_loss(second)
        return {"total": first_loss + second_loss, "p_theta": first_loss, "q_v": second_loss}

    def sample(
        self,
        sample_count: int,
        device: torch.device,
        guidance_scale: float,
        inequality_weight: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """Run Algorithm 5 and return physical common states and PGE count."""

        shape = (sample_count, 2 * self.bus_count)
        first = torch.randn(shape, device=device)
        second = torch.randn(shape, device=device)
        physics_evaluations = 0
        for step in reversed(range(self.p_theta.steps)):
            timestep = torch.full((sample_count,), step, device=device, dtype=torch.long)
            if guidance_scale > 0.0:
                first = first.detach().requires_grad_(True)
                second = second.detach().requires_grad_(True)
            with torch.set_grad_enabled(guidance_scale > 0.0):
                first_noise = self.p_theta.denoiser(first, timestep)
                second_noise = self.q_v.denoiser(second, timestep)
                first_clean = self.p_theta.predict_clean(first, timestep, first_noise)
                second_clean = self.q_v.predict_clean(second, timestep, second_noise)
                if guidance_scale > 0.0:
                    normalized_common = self.combine(first_clean, second_clean)
                    physical_common = self.scaler.inverse(normalized_common)
                    residual = equality_loss(physical_common, self.grid)
                    residual = residual + inequality_weight * inequality_loss(
                        physical_common, self.grid
                    )
                    gradient_first, gradient_second = torch.autograd.grad(
                        residual, (first, second)
                    )
                    # Eq. (23) uses the gradient with respect to x_t to correct
                    # the clean estimate before adding the next noise level.
                    first_clean = first_clean - guidance_scale * gradient_first
                    second_clean = second_clean - guidance_scale * gradient_second
                    physics_evaluations += 1
            with torch.no_grad():
                first = self.p_theta.posterior_sample(
                    first.detach(), first_clean.detach(), timestep
                )
                second = self.q_v.posterior_sample(
                    second.detach(), second_clean.detach(), timestep
                )
        normalized = self.combine(first, second)
        return self.scaler.inverse(normalized), physics_evaluations
