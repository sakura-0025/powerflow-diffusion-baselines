"""Train-only fitted scaling used by the three baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TensorMinMaxScaler:
    """Feature-wise min-max scaling with a serializable state."""

    minimum: torch.Tensor
    maximum: torch.Tensor
    low: float = -1.0
    high: float = 1.0

    @classmethod
    def fit(
        cls, values: torch.Tensor, low: float = -1.0, high: float = 1.0
    ) -> "TensorMinMaxScaler":
        if values.ndim < 2:
            raise ValueError("Scaler expects a sample dimension and feature dimensions.")
        minimum = values.amin(dim=0)
        maximum = values.amax(dim=0)
        return cls(minimum, maximum, low, high)

    @property
    def span(self) -> torch.Tensor:
        return (self.maximum - self.minimum).clamp_min(1.0e-8)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        unit = (values - self.minimum.to(values.device)) / self.span.to(values.device)
        return unit * (self.high - self.low) + self.low

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        unit = (values - self.low) / (self.high - self.low)
        return unit * self.span.to(values.device) + self.minimum.to(values.device)

    def state_dict(self) -> dict[str, object]:
        return {
            "minimum": self.minimum.cpu(),
            "maximum": self.maximum.cpu(),
            "low": self.low,
            "high": self.high,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "TensorMinMaxScaler":
        return cls(
            torch.as_tensor(state["minimum"]),
            torch.as_tensor(state["maximum"]),
            float(state["low"]),
            float(state["high"]),
        )

