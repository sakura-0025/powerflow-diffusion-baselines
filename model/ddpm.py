"""Ordinary noise-prediction DDPM and the shared vector denoiser."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timestep.float().unsqueeze(-1) * frequencies
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.channels:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, time_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden + time_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.norm(values + self.layers(torch.cat([values, time], dim=-1)))


class VectorDenoiser(nn.Module):
    """Configurable residual MLP used when the paper leaves architecture missing."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 512,
        layers: int = 6,
        time_channels: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_channels),
            nn.Linear(time_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU())
        self.blocks = nn.ModuleList([ResidualBlock(hidden, time_channels) for _ in range(layers)])
        self.output_layer = nn.Sequential(nn.SiLU(), nn.Linear(hidden, input_dim))

    def forward(self, values: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_shape = values.shape
        flat = values.reshape(values.shape[0], -1)
        if flat.shape[1] != self.input_dim:
            raise ValueError("Input does not match denoiser dimension.")
        hidden = self.input_layer(flat)
        time = self.time_encoder(timestep)
        for block in self.blocks:
            hidden = block(hidden, time)
        return self.output_layer(hidden).reshape(original_shape)


@dataclass(frozen=True)
class DiffusionConfig:
    steps: int = 200
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2


class GaussianDiffusion(nn.Module):
    """DDPM equations shared by the three implementations."""

    def __init__(
        self,
        denoiser: nn.Module,
        config: DiffusionConfig,
        alpha_bars: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        if alpha_bars is None:
            betas = torch.linspace(config.beta_start, config.beta_end, config.steps)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
        else:
            alpha_bars = alpha_bars.detach().float()
            if len(alpha_bars) != config.steps:
                raise ValueError("Learned alpha_bars length does not match diffusion steps.")
            previous = torch.cat([torch.ones(1), alpha_bars[:-1]])
            alphas = (alpha_bars / previous).clamp(1.0e-5, 0.999999)
            betas = 1.0 - alphas
        previous_alpha_bars = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior_variance = betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        posterior_mean_x0 = betas * previous_alpha_bars.sqrt() / (1.0 - alpha_bars)
        posterior_mean_xt = alphas.sqrt() * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1.0e-20))
        self.register_buffer("posterior_mean_x0", posterior_mean_x0)
        self.register_buffer("posterior_mean_xt", posterior_mean_xt)

    @property
    def steps(self) -> int:
        return len(self.betas)

    @staticmethod
    def extract(values: torch.Tensor, timestep: torch.Tensor, ndim: int) -> torch.Tensor:
        return values[timestep].view((len(timestep),) + (1,) * (ndim - 1))

    def q_sample(self, clean: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.extract(self.alpha_bars, timestep, clean.ndim)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def predict_clean(
        self, noisy: torch.Tensor, timestep: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.extract(self.alpha_bars, timestep, noisy.ndim)
        return (noisy - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1.0e-8)

    def noise_loss(self, clean: torch.Tensor) -> torch.Tensor:
        timestep = torch.randint(self.steps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timestep, noise)
        return torch.mean((self.denoiser(noisy, timestep) - noise).square())

    def posterior_sample(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.extract(self.posterior_mean_x0, timestep, noisy.ndim) * clean
        mean += self.extract(self.posterior_mean_xt, timestep, noisy.ndim) * noisy
        noise = torch.randn_like(noisy)
        nonzero = (timestep > 0).view((len(timestep),) + (1,) * (noisy.ndim - 1))
        return mean + nonzero * self.extract(self.posterior_variance, timestep, noisy.ndim).sqrt() * noise

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        noisy = torch.randn(shape, device=device)
        for step in reversed(range(self.steps)):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = self.denoiser(noisy, timestep)
            clean = self.predict_clean(noisy, timestep, predicted_noise)
            noisy = self.posterior_sample(noisy, clean, timestep)
        return noisy
