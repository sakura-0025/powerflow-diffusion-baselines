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


def guidance_step_enabled(
    step: int,
    alpha_bar: float,
    guidance_scale: float,
    guidance_last_steps: int | None,
    guidance_alpha_bar_min: float,
) -> bool:
    """Return whether the project extension applies guidance at this timestep.

    Leaving ``guidance_last_steps`` unset and ``guidance_alpha_bar_min`` at
    zero reproduces the paper-aligned all-step guidance behavior.
    """

    return bool(
        guidance_scale > 0.0
        and (guidance_last_steps is None or step < guidance_last_steps)
        and alpha_bar >= guidance_alpha_bar_min
    )


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
        guidance_last_steps: int | None = None,
        guidance_alpha_bar_min: float = 0.0,
        guidance_residual_threshold: float = 0.0,
        normalize_guidance_gradient: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Run Algorithm 5 with optional project-controlled sparse guidance.

        The paper-aligned baseline is unchanged when the optional controls use
        their defaults. Diagnostics report whether a single correction moves
        the predicted clean sample toward a lower constraint energy.
        """

        shape = (sample_count, 2 * self.bus_count)
        first = torch.randn(shape, device=device)
        second = torch.randn(shape, device=device)
        physics_evaluations = 0
        guidance_step_calls = 0
        physics_gradient_batch_calls = 0
        trace: list[dict[str, float | int]] = []
        for step in reversed(range(self.p_theta.steps)):
            timestep = torch.full((sample_count,), step, device=device, dtype=torch.long)
            alpha_bar = float(self.p_theta.alpha_bars[step].detach().cpu())
            step_guidance = guidance_step_enabled(
                step,
                alpha_bar,
                guidance_scale,
                guidance_last_steps,
                guidance_alpha_bar_min,
            )
            if step_guidance:
                first = first.detach().requires_grad_(True)
                second = second.detach().requires_grad_(True)
            with torch.set_grad_enabled(step_guidance):
                first_noise = self.p_theta.denoiser(first, timestep)
                second_noise = self.q_v.denoiser(second, timestep)
                first_clean = self.p_theta.predict_clean(first, timestep, first_noise)
                second_clean = self.q_v.predict_clean(second, timestep, second_noise)
                if step_guidance:
                    normalized_common = self.combine(first_clean, second_clean)
                    physical_common = self.scaler.inverse(normalized_common)
                    residual_per_sample = equality_loss(physical_common, self.grid)
                    residual_per_sample = residual_per_sample + inequality_weight * inequality_loss(
                        physical_common, self.grid
                    )
                    active = residual_per_sample > guidance_residual_threshold
                    active_count = int(active.sum().detach().cpu())
                    raw_gradient_norm_mean = 0.0
                    applied_gradient_norm_mean = 0.0
                    relative_update_norm_mean = 0.0
                    residual_before_mean = 0.0
                    residual_after_mean = 0.0
                    residual_reduction_fraction_mean = 0.0
                    harmful_update_fraction = 0.0
                    if active_count:
                        # Summing independent per-sample residuals gives every
                        # sample its own paper-defined gradient. A batch mean
                        # would silently divide guidance strength by batch size.
                        gradient_first, gradient_second = torch.autograd.grad(
                            (residual_per_sample * active).sum(), (first, second)
                        )
                        physics_gradient_batch_calls += 1
                        raw_norm = torch.sqrt(
                            gradient_first.square().sum(dim=1)
                            + gradient_second.square().sum(dim=1)
                        )
                        if normalize_guidance_gradient:
                            denominator = raw_norm.clamp_min(1.0e-12).unsqueeze(1)
                            gradient_first = gradient_first / denominator
                            gradient_second = gradient_second / denominator
                        applied_norm = torch.sqrt(
                            gradient_first.square().sum(dim=1)
                            + gradient_second.square().sum(dim=1)
                        )
                        active_float = active.to(first_clean.dtype).unsqueeze(1)
                        gradient_first = gradient_first * active_float
                        gradient_second = gradient_second * active_float
                        first_clean = first_clean - guidance_scale * gradient_first
                        second_clean = second_clean - guidance_scale * gradient_second

                        with torch.no_grad():
                            corrected_physical = self.scaler.inverse(
                                self.combine(first_clean, second_clean)
                            )
                            residual_after = equality_loss(corrected_physical, self.grid)
                            residual_after = residual_after + inequality_weight * inequality_loss(
                                corrected_physical, self.grid
                            )
                            before_active = residual_per_sample[active].detach()
                            after_active = residual_after[active]
                            clean_norm = torch.sqrt(
                                first_clean.square().sum(dim=1)
                                + second_clean.square().sum(dim=1)
                            ).clamp_min(1.0e-12)
                            residual_before_mean = float(before_active.mean().cpu())
                            residual_after_mean = float(after_active.mean().cpu())
                            residual_reduction_fraction_mean = float(
                                (
                                    (before_active - after_active)
                                    / before_active.clamp_min(1.0e-12)
                                )
                                .mean()
                                .cpu()
                            )
                            harmful_update_fraction = float(
                                (after_active > before_active).float().mean().cpu()
                            )
                            raw_gradient_norm_mean = float(raw_norm[active].mean().cpu())
                            applied_gradient_norm_mean = float(
                                applied_norm[active].mean().cpu()
                            )
                            relative_update_norm_mean = float(
                                (
                                    guidance_scale
                                    * applied_norm[active]
                                    / clean_norm[active]
                                )
                                .mean()
                                .cpu()
                            )
                        physics_evaluations += active_count
                    guidance_step_calls += 1
                    trace.append(
                        {
                            "step": int(step),
                            "alpha_bar": alpha_bar,
                            "active_samples": active_count,
                            "total_samples": int(sample_count),
                            "residual_before_mean": residual_before_mean,
                            "residual_after_mean": residual_after_mean,
                            "residual_reduction_fraction_mean": (
                                residual_reduction_fraction_mean
                            ),
                            "harmful_update_fraction": harmful_update_fraction,
                            "raw_gradient_norm_mean": raw_gradient_norm_mean,
                            "applied_gradient_norm_mean": applied_gradient_norm_mean,
                            "relative_update_norm_mean": relative_update_norm_mean,
                        }
                    )
            with torch.no_grad():
                first = self.p_theta.posterior_sample(
                    first.detach(), first_clean.detach(), timestep
                )
                second = self.q_v.posterior_sample(
                    second.detach(), second_clean.detach(), timestep
                )
        normalized = self.combine(first, second)
        return self.scaler.inverse(normalized), {
            "physics_gradient_evaluations_total": int(physics_evaluations),
            "guidance_step_calls": int(guidance_step_calls),
            "physics_gradient_batch_calls": int(physics_gradient_batch_calls),
            "trace": trace,
        }
