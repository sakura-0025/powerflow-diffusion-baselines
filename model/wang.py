"""Reconstruction of Wang et al.'s physics-informed DDPM.

Source-stated components:
* seven hidden layers [256,128,64,32,64,128,256];
* an auxiliary 512/256 network that learns a 200-step schedule;
* the hinge physics loss max(R(x_t)-gamma_t, 0), with eta=1;
* Wang's separate load/generation state representation scaled to [0,1].

The paper does not publish code and does not explain how schedule monotonicity
is guaranteed.  ``WangScheduleNetwork`` therefore predicts valid beta values
and obtains alpha_bar by cumulative product.  This compact helper is explicitly
an inferred, replaceable reconstruction.
"""

from __future__ import annotations

import torch
from torch import nn

from .ddpm import DiffusionConfig, GaussianDiffusion, SinusoidalTimeEmbedding
from .physics import GridPhysics, mean_complex_imbalance, wang_to_common
from .scaling import TensorMinMaxScaler


class TimedLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, time_dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(input_dim, output_dim)
        self.time = nn.Linear(time_dim, output_dim)
        # This time-dependent gate is the smallest replaceable interpretation
        # of the paper's unspecified "attention units".
        self.gate = nn.Linear(time_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.value(values) + self.time(time))
        return self.norm(hidden * torch.sigmoid(self.gate(time)))


class WangDenoiser(nn.Module):
    """Symmetric seven-layer vector U-Net described in Wang Sec. IV-B2."""

    def __init__(self, input_dim: int, time_dim: int = 128, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.sigmoid_output = sigmoid_output
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.down1 = TimedLayer(input_dim, 256, time_dim)
        self.down2 = TimedLayer(256, 128, time_dim)
        self.down3 = TimedLayer(128, 64, time_dim)
        self.middle = TimedLayer(64, 32, time_dim)
        self.up1 = TimedLayer(32 + 64, 64, time_dim)
        self.up2 = TimedLayer(64 + 128, 128, time_dim)
        self.up3 = TimedLayer(128 + 256, 256, time_dim)
        self.output = nn.Linear(256, input_dim)

    def forward(self, values: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(values.shape[0], -1)
        time = self.time_encoder(timestep)
        d1 = self.down1(flat, time)
        d2 = self.down2(d1, time)
        d3 = self.down3(d2, time)
        middle = self.middle(d3, time)
        u1 = self.up1(torch.cat([middle, d3], dim=-1), time)
        u2 = self.up2(torch.cat([u1, d2], dim=-1), time)
        u3 = self.up3(torch.cat([u2, d1], dim=-1), time)
        output = self.output(u3)
        # Eq. (5) predicts Gaussian noise, which can be negative, while the
        # prose says Sigmoid. Linear is the runnable default; strict wording is
        # available for protocol sensitivity through this flag.
        if self.sigmoid_output:
            output = torch.sigmoid(output)
        return output.reshape_as(values)


class WangScheduleNetwork(nn.Module):
    """Auxiliary 512/256 schedule network from Wang Sec. III-B2."""

    def __init__(self, input_dim: int, steps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, steps),
        )

    def forward(self, clean_batch: torch.Tensor) -> torch.Tensor:
        # Paper: one unified schedule is obtained by averaging outputs over
        # real inputs. Predicting betas rather than unconstrained alpha_bars is
        # the documented inference that guarantees a valid monotone schedule.
        raw = self.network(clean_batch.reshape(len(clean_batch), -1)).mean(dim=0)
        betas = self.beta_start + torch.sigmoid(raw) * (self.beta_end - self.beta_start)
        return torch.cumprod(1.0 - betas, dim=0)


class WangPhysicsInformedDDPM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        diffusion_config: DiffusionConfig,
        scaler: TensorMinMaxScaler,
        grid: GridPhysics,
        load_buses: torch.Tensor,
        generator_buses: torch.Tensor,
        alpha_bars: torch.Tensor | None = None,
        physics_weight: float = 1.0,
        gamma_terminal: float = 2.75,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.scaler = scaler
        self.grid = grid
        self.register_buffer("load_buses", load_buses.long())
        self.register_buffer("generator_buses", generator_buses.long())
        self.physics_weight = physics_weight
        self.gamma_terminal = gamma_terminal
        self.diffusion = GaussianDiffusion(
            WangDenoiser(input_dim, sigmoid_output=sigmoid_output),
            diffusion_config,
            alpha_bars=alpha_bars,
        )

    def decode_common(self, normalized_wang: torch.Tensor) -> torch.Tensor:
        native = self.scaler.inverse(normalized_wang)
        return wang_to_common(
            native,
            self.load_buses,
            self.generator_buses,
            self.grid.ybus.shape[0],
            self.grid.base_mva,
        )

    def training_losses(self, normalized_clean: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = len(normalized_clean)
        timestep = torch.randint(self.diffusion.steps, (batch,), device=normalized_clean.device)
        noise = torch.randn_like(normalized_clean)
        noisy = self.diffusion.q_sample(normalized_clean, timestep, noise)
        predicted = self.diffusion.denoiser(noisy, timestep)
        noise_loss = (predicted - noise).square().mean()
        imbalance = mean_complex_imbalance(self.decode_common(noisy), self.grid)
        gamma = self.gamma_terminal * (timestep.float() + 1.0) / self.diffusion.steps
        physics_loss = torch.relu(imbalance - gamma).mean()
        return {
            "total": noise_loss + self.physics_weight * physics_loss,
            "noise": noise_loss,
            "physics": physics_loss,
        }


def estimate_terminal_imbalance(
    scaler: TensorMinMaxScaler,
    grid: GridPhysics,
    load_buses: torch.Tensor,
    generator_buses: torch.Tensor,
    input_dim: int,
    samples: int = 1024,
) -> float:
    """Estimate gamma_T from Gaussian noise as done empirically in the paper."""

    with torch.no_grad():
        noisy = torch.randn(samples, input_dim, device=grid.ybus.device)
        native = scaler.inverse(noisy)
        common = wang_to_common(
            native,
            load_buses.to(noisy.device),
            generator_buses.to(noisy.device),
            grid.ybus.shape[0],
            grid.base_mva,
        )
        return float(mean_complex_imbalance(common, grid).mean().cpu())

