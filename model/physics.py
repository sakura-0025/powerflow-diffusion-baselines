"""Differentiable AC power-flow constraints shared by every evaluator.

All equality residuals are computed in per-unit from the complex nodal
admittance matrix.  Keeping one independent implementation prevents each
baseline from being judged by a different definition of feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GridPhysics:
    ybus: torch.Tensor
    yf: torch.Tensor
    yt: torch.Tensor
    branch_from: torch.Tensor
    branch_to: torch.Tensor
    branch_rate_pu: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    base_mva: float

    def to(self, device: torch.device | str) -> "GridPhysics":
        return GridPhysics(
            self.ybus.to(device),
            self.yf.to(device),
            self.yt.to(device),
            self.branch_from.to(device),
            self.branch_to.to(device),
            self.branch_rate_pu.to(device),
            self.lower.to(device),
            self.upper.to(device),
            self.base_mva,
        )


def physics_from_archive(archive: dict[str, object]) -> GridPhysics:
    def complex_matrix(real_key: str, imag_key: str) -> torch.Tensor:
        return torch.complex(
            torch.from_numpy(np.asarray(archive[real_key], dtype=np.float32)),
            torch.from_numpy(np.asarray(archive[imag_key], dtype=np.float32)),
        )

    metadata = archive["metadata"]
    assert isinstance(metadata, dict)
    return GridPhysics(
        ybus=complex_matrix("ybus_real", "ybus_imag"),
        yf=complex_matrix("yf_real", "yf_imag"),
        yt=complex_matrix("yt_real", "yt_imag"),
        branch_from=torch.from_numpy(np.asarray(archive["branch_from"], dtype=np.int64)),
        branch_to=torch.from_numpy(np.asarray(archive["branch_to"], dtype=np.int64)),
        branch_rate_pu=torch.from_numpy(
            np.asarray(archive["branch_rate_pu"], dtype=np.float32)
        ),
        lower=torch.from_numpy(np.asarray(archive["common_lower"], dtype=np.float32)),
        upper=torch.from_numpy(np.asarray(archive["common_upper"], dtype=np.float32)),
        base_mva=float(metadata["base_mva"]),
    )


def ac_power_balance(state: torch.Tensor, grid: GridPhysics) -> torch.Tensor:
    """Return [batch,bus,2] residuals for active and reactive balance."""

    if state.ndim != 3 or state.shape[-1] != 4:
        raise ValueError("Common state must have shape [batch, bus, 4].")
    injection = torch.complex(state[..., 0], state[..., 1])
    voltage = torch.polar(state[..., 2], state[..., 3])
    calculated = voltage * torch.conj(torch.einsum("ij,bj->bi", grid.ybus, voltage))
    mismatch = injection - calculated
    return torch.stack([mismatch.real, mismatch.imag], dim=-1)


def line_apparent_power(state: torch.Tensor, grid: GridPhysics) -> tuple[torch.Tensor, torch.Tensor]:
    """Return from- and to-end apparent branch powers in per-unit."""

    voltage = torch.polar(state[..., 2], state[..., 3])
    current_from = torch.einsum("lj,bj->bl", grid.yf, voltage)
    current_to = torch.einsum("lj,bj->bl", grid.yt, voltage)
    from_voltage = voltage[:, grid.branch_from]
    to_voltage = voltage[:, grid.branch_to]
    return torch.abs(from_voltage * torch.conj(current_from)), torch.abs(
        to_voltage * torch.conj(current_to)
    )


def equality_loss(state: torch.Tensor, grid: GridPhysics) -> torch.Tensor:
    return ac_power_balance(state, grid).square().mean()


def inequality_loss(state: torch.Tensor, grid: GridPhysics) -> torch.Tensor:
    """Squared hinge residual for state bounds and finite branch ratings."""

    lower = grid.lower.to(state.device)
    upper = grid.upper.to(state.device)
    bound_violation = torch.relu(lower - state).square() + torch.relu(state - upper).square()
    from_flow, to_flow = line_apparent_power(state, grid)
    limits = grid.branch_rate_pu.to(state.device)
    finite = torch.isfinite(limits)
    thermal = state.new_zeros(())
    if bool(finite.any()):
        thermal = (
            torch.relu(from_flow[:, finite] - limits[finite]).square().mean()
            + torch.relu(to_flow[:, finite] - limits[finite]).square().mean()
        )
    return bound_violation.mean() + thermal


def mean_complex_imbalance(state: torch.Tensor, grid: GridPhysics) -> torch.Tensor:
    """Wang Eq. (7): mean magnitude of complex nodal imbalance per sample."""

    residual = ac_power_balance(state, grid)
    return torch.linalg.vector_norm(residual, dim=-1).mean(dim=-1)


def wang_to_common(
    wang_state: torch.Tensor,
    load_buses: torch.Tensor,
    generator_buses: torch.Tensor,
    bus_count: int,
    base_mva: float,
) -> torch.Tensor:
    """Decode Wang's paper-specific vector into [p,q,v,theta] at every bus."""

    batch = wang_state.shape[0]
    load_count = len(load_buses)
    generator_count = len(generator_buses)
    cursor = 0
    p_load = wang_state[:, cursor : cursor + load_count]
    cursor += load_count
    q_load = wang_state[:, cursor : cursor + load_count]
    cursor += load_count
    vm = wang_state[:, cursor : cursor + bus_count]
    cursor += bus_count
    theta = wang_state[:, cursor : cursor + bus_count]
    cursor += bus_count
    p_gen = wang_state[:, cursor : cursor + generator_count]
    cursor += generator_count
    q_gen = wang_state[:, cursor : cursor + generator_count]
    if cursor + generator_count != wang_state.shape[1]:
        raise ValueError("Wang state dimension is inconsistent with bus masks.")

    p = wang_state.new_zeros((batch, bus_count))
    q = wang_state.new_zeros((batch, bus_count))
    p[:, load_buses] -= p_load / base_mva
    q[:, load_buses] -= q_load / base_mva
    p[:, generator_buses] += p_gen / base_mva
    q[:, generator_buses] += q_gen / base_mva
    return torch.stack([p, q, vm, theta], dim=-1)

