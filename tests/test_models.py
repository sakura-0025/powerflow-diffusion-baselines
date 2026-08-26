from __future__ import annotations

import torch

from model.ddpm import DiffusionConfig, GaussianDiffusion, VectorDenoiser
from model.physics import GridPhysics, ac_power_balance, equality_loss, wang_to_common
from model.wang import WangDenoiser, WangScheduleNetwork


def two_bus_grid() -> GridPhysics:
    admittance = torch.tensor([[1.0 - 2.0j, -1.0 + 2.0j], [-1.0 + 2.0j, 1.0 - 2.0j]])
    empty = torch.empty((0, 2), dtype=torch.complex64)
    return GridPhysics(
        ybus=admittance.to(torch.complex64),
        yf=empty,
        yt=empty,
        branch_from=torch.empty(0, dtype=torch.long),
        branch_to=torch.empty(0, dtype=torch.long),
        branch_rate_pu=torch.empty(0),
        lower=torch.full((2, 4), -10.0),
        upper=torch.full((2, 4), 10.0),
        base_mva=100.0,
    )


def test_balanced_flat_two_bus_state_has_zero_residual() -> None:
    state = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    residual = ac_power_balance(state, two_bus_grid())
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-6)


def test_wang_state_decodes_to_common_injections() -> None:
    # one load bus, one generator bus, then V/theta for both buses
    native = torch.tensor([[10.0, 5.0, 1.0, 0.98, 0.0, -0.02, 12.0, 6.0]])
    common = wang_to_common(
        native,
        torch.tensor([1]),
        torch.tensor([0]),
        bus_count=2,
        base_mva=100.0,
    )
    assert common.shape == (1, 2, 4)
    assert torch.allclose(common[0, :, 0], torch.tensor([0.12, -0.10]))
    assert torch.allclose(common[0, :, 1], torch.tensor([0.06, -0.05]))


def test_denoisers_and_diffusion_preserve_shapes() -> None:
    denoiser = VectorDenoiser(8, hidden=32, layers=2, time_channels=16)
    diffusion = GaussianDiffusion(denoiser, DiffusionConfig(steps=4))
    clean = torch.randn(3, 2, 4)
    assert diffusion.noise_loss(clean).ndim == 0
    generated = diffusion.sample((3, 2, 4), torch.device("cpu"))
    assert generated.shape == clean.shape

    wang = WangDenoiser(8, time_dim=16)
    assert wang(torch.randn(3, 8), torch.tensor([0, 1, 2])).shape == (3, 8)


def test_wang_schedule_is_monotone() -> None:
    schedule = WangScheduleNetwork(8, steps=10, beta_start=1.0e-4, beta_end=0.1)
    alpha_bars = schedule(torch.randn(5, 8))
    assert alpha_bars.shape == (10,)
    assert torch.all(alpha_bars[1:] < alpha_bars[:-1])


def test_constraint_gradient_is_invariant_to_batch_size() -> None:
    """Paper guidance lambda must not change when sampling batch size changes."""

    single = torch.tensor(
        [[[0.10, 0.02, 1.0, 0.0], [-0.08, -0.01, 0.98, -0.03]]],
        requires_grad=True,
    )
    single_gradient = torch.autograd.grad(
        equality_loss(single, two_bus_grid()).sum(), single
    )[0]

    repeated = single.detach().repeat(5, 1, 1).requires_grad_(True)
    repeated_gradient = torch.autograd.grad(
        equality_loss(repeated, two_bus_grid()).sum(), repeated
    )[0]
    assert torch.allclose(single_gradient[0], repeated_gradient[0], atol=1.0e-6)
