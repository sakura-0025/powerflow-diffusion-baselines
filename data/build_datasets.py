"""Build AC-OPF datasets for IEEE 14-, 30- and 118-bus systems.

The two papers use random load perturbations followed by a feasible AC-OPF.
This implementation exposes both source protocols and a common comparison
protocol.  It never silently replaces a failed OPF sample: failed attempts are
counted and discarded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from tqdm import tqdm


CASE_FACTORIES = {
    "ieee14": pn.case14,
    "ieee30": pn.case30,
    "ieee118": pn.case118,
}


@dataclass(frozen=True)
class BuildConfig:
    """Configuration recorded verbatim in the dataset metadata."""

    case: str
    samples: int = 1_000
    protocol: str = "common"
    seed: int = 2026
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    max_attempt_factor: int = 20
    retry_flat: bool = False

    def load_bounds(self) -> tuple[float, float]:
        # Hoseinpour Sec. VI-A states [0.8, 1.0]; Wang Sec. IV-A states [0.8, 1.2].
        return (0.8, 1.0) if self.protocol == "hoseinpour" else (0.8, 1.2)

    @property
    def perturb_costs(self) -> bool:
        # Wang additionally perturbs generator costs in [50%, 150%].
        return self.protocol == "wang"


def _factory(case: str) -> pp.pandapowerNet:
    try:
        return CASE_FACTORIES[case]()
    except KeyError as error:
        raise ValueError(f"Unsupported case {case!r}; choose {tuple(CASE_FACTORIES)}") from error


def _bus_sum(bus_count: int, buses: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros(bus_count, dtype=np.float64)
    if len(buses):
        np.add.at(result, buses.astype(np.int64), values.astype(np.float64))
    return result


def _extract_sample(
    net: pp.pandapowerNet,
    load_buses: np.ndarray,
    generator_buses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return common [B,4] and Wang-style flat states from one solved OPF."""

    bus_count = len(net.bus)
    if not np.array_equal(net.bus.index.to_numpy(), np.arange(bus_count)):
        raise ValueError("Standard-case bus indices must be contiguous from zero.")

    p_load = _bus_sum(bus_count, net.load.bus.to_numpy(), net.res_load.p_mw.to_numpy())
    q_load = _bus_sum(bus_count, net.load.bus.to_numpy(), net.res_load.q_mvar.to_numpy())
    p_gen = np.zeros(bus_count, dtype=np.float64)
    q_gen = np.zeros(bus_count, dtype=np.float64)
    if len(net.gen):
        p_gen += _bus_sum(bus_count, net.gen.bus.to_numpy(), net.res_gen.p_mw.to_numpy())
        q_gen += _bus_sum(bus_count, net.gen.bus.to_numpy(), net.res_gen.q_mvar.to_numpy())
    if len(net.sgen):
        p_gen += _bus_sum(bus_count, net.sgen.bus.to_numpy(), net.res_sgen.p_mw.to_numpy())
        q_gen += _bus_sum(bus_count, net.sgen.bus.to_numpy(), net.res_sgen.q_mvar.to_numpy())
    if len(net.ext_grid):
        p_gen += _bus_sum(
            bus_count, net.ext_grid.bus.to_numpy(), net.res_ext_grid.p_mw.to_numpy()
        )
        q_gen += _bus_sum(
            bus_count, net.ext_grid.bus.to_numpy(), net.res_ext_grid.q_mvar.to_numpy()
        )

    vm = net.res_bus.vm_pu.to_numpy(dtype=np.float64)
    theta = np.deg2rad(net.res_bus.va_degree.to_numpy(dtype=np.float64))
    base_mva = float(net.sn_mva)
    common = np.stack(
        [(p_gen - p_load) / base_mva, (q_gen - q_load) / base_mva, vm, theta],
        axis=-1,
    )

    # These bus lists come from the network element tables, not from the
    # solved numerical values.  Their dimensions therefore remain fixed even
    # when an OPF dispatch happens to set one generator exactly to zero.
    wang = np.concatenate(
        [
            p_load[load_buses],
            q_load[load_buses],
            vm,
            theta,
            p_gen[generator_buses],
            q_gen[generator_buses],
        ]
    )
    return common, wang


def _network_arrays(net: pp.pandapowerNet) -> dict[str, np.ndarray]:
    """Extract matrices needed for independent differentiable physics checks."""

    ppc = net._ppc
    ybus, yf, yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    branch = ppc["branch"]
    rate = branch[:, 5].real / float(ppc["baseMVA"])
    # MATPOWER uses zero to denote an unconstrained thermal limit.
    rate = np.where(rate > 0.0, rate, np.inf)
    return {
        "ybus_real": ybus.toarray().real.astype(np.float64),
        "ybus_imag": ybus.toarray().imag.astype(np.float64),
        "yf_real": yf.toarray().real.astype(np.float64),
        "yf_imag": yf.toarray().imag.astype(np.float64),
        "yt_real": yt.toarray().real.astype(np.float64),
        "yt_imag": yt.toarray().imag.astype(np.float64),
        "branch_from": branch[:, 0].real.astype(np.int64),
        "branch_to": branch[:, 1].real.astype(np.int64),
        "branch_rate_pu": rate.astype(np.float64),
    }


def _native_bus_masks(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    load_mask = np.zeros(len(net.bus), dtype=bool)
    load_mask[net.load.bus.to_numpy(dtype=np.int64)] = True
    generator_mask = np.zeros(len(net.bus), dtype=bool)
    generator_mask[net.gen.bus.to_numpy(dtype=np.int64)] = True
    generator_mask[net.ext_grid.bus.to_numpy(dtype=np.int64)] = True
    if len(net.sgen):
        generator_mask[net.sgen.bus.to_numpy(dtype=np.int64)] = True
    return np.flatnonzero(load_mask), np.flatnonzero(generator_mask)


def _run_ac_opf(
    net: pp.pandapowerNet, retry_flat: bool
) -> tuple[bool, str, int]:
    """Solve one sample with a documented warm-start/fallback sequence.

    PYPOWER convergence can depend on initialization, especially for IEEE 30.
    Retrying the *same* sampled operating point avoids replacing a numerical
    failure with a different, easier random point.  The successful mode and
    total solver calls are recorded for the dataset manifest.
    """

    calls = 0
    initializations = ("pf", "flat") if retry_flat else ("pf",)
    for initialization in initializations:
        calls += 1
        try:
            pp.runopp(
                net,
                calculate_voltage_angles=True,
                init=initialization,
                suppress_warnings=True,
                verbose=False,
            )
            if bool(net.OPF_converged):
                return True, initialization, calls
        except Exception:
            # A second documented initialization is still attempted below.
            pass
    return False, "failed", calls


def build_dataset(output: Path, config: BuildConfig) -> None:
    if config.samples < 3:
        raise ValueError("At least three samples are required for disjoint splits.")
    if config.protocol not in {"common", "wang", "hoseinpour"}:
        raise ValueError("protocol must be common, wang, or hoseinpour.")
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave a test split.")

    rng = np.random.default_rng(config.seed)
    build_started = time.perf_counter()
    net = _factory(config.case)
    nominal_p = net.load.p_mw.to_numpy(dtype=np.float64).copy()
    nominal_q = net.load.q_mvar.to_numpy(dtype=np.float64).copy()
    original_cost = net.poly_cost.copy(deep=True) if len(net.poly_cost) else None
    load_buses, generator_buses = _native_bus_masks(net)
    low, high = config.load_bounds()
    common_samples: list[np.ndarray] = []
    wang_samples: list[np.ndarray] = []
    accepted_load_scales: list[np.ndarray] = []
    accepted_system_scales: list[float] = []
    rejected_system_scales: list[float] = []
    accepted_cost_scales: list[np.ndarray] = []
    failures = 0
    attempts = 0
    opf_calls = 0
    initialization_counts = {"pf": 0, "flat": 0}
    cost_columns = tuple(
        column
        for column in ("cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2")
        if original_cost is not None and column in net.poly_cost
    )

    progress = tqdm(total=config.samples, desc=f"build {config.case}")
    while len(common_samples) < config.samples:
        attempts += 1
        if attempts > config.samples * config.max_attempt_factor:
            raise RuntimeError(
                f"Only {len(common_samples)} feasible samples after {attempts} attempts."
            )
        # A shared scale preserves each load's nominal power factor.
        scale = rng.uniform(low, high, size=len(net.load))
        system_scale = float(np.sum(nominal_p * scale) / np.sum(nominal_p))
        net.load.loc[:, "p_mw"] = nominal_p * scale
        net.load.loc[:, "q_mvar"] = nominal_q * scale
        cost_scale = np.ones((len(net.poly_cost), len(cost_columns)), dtype=np.float64)
        if original_cost is not None:
            # Only numeric cost coefficients change. Reassigning identifier
            # columns would coerce pandas integer dtypes and is unnecessary.
            for column_index, column in enumerate(cost_columns):
                factor = (
                    rng.uniform(0.5, 1.5, size=len(original_cost))
                    if config.perturb_costs
                    else np.ones(len(original_cost))
                )
                cost_scale[:, column_index] = factor
                net.poly_cost.loc[:, column] = original_cost[column].to_numpy() * factor
        try:
            converged, initialization, calls = _run_ac_opf(net, config.retry_flat)
            opf_calls += calls
            if not converged:
                raise RuntimeError("AC-OPF did not converge")
            common, wang = _extract_sample(net, load_buses, generator_buses)
            if not np.isfinite(common).all() or not np.isfinite(wang).all():
                raise RuntimeError("Non-finite OPF result")
        except Exception:
            failures += 1
            rejected_system_scales.append(system_scale)
            continue
        initialization_counts[initialization] += 1
        common_samples.append(common.astype(np.float32))
        wang_samples.append(wang.astype(np.float32))
        accepted_load_scales.append(scale.astype(np.float32))
        accepted_system_scales.append(system_scale)
        accepted_cost_scales.append(cost_scale.astype(np.float32))
        progress.update(1)
    progress.close()

    common_array = np.stack(common_samples)
    wang_array = np.stack(wang_samples)
    load_scale_array = np.stack(accepted_load_scales)
    system_scale_array = np.asarray(accepted_system_scales, dtype=np.float32)
    rejected_system_scale_array = np.asarray(rejected_system_scales, dtype=np.float32)
    cost_scale_array = np.stack(accepted_cost_scales)
    order = rng.permutation(config.samples)
    split = np.full(config.samples, 2, dtype=np.int8)
    # Preserve non-empty train/validation/test sets even for the three-sample
    # mechanism check; at formal sizes this reduces exactly to the requested
    # fractions (for example 80/10/10 for 100 samples).
    train_count = min(
        max(1, int(round(config.samples * config.train_fraction))),
        config.samples - 2,
    )
    validation_count = min(
        max(1, int(round(config.samples * config.validation_fraction))),
        config.samples - train_count - 1,
    )
    train_end = train_count
    validation_end = train_count + validation_count
    split[order[:train_end]] = 0
    split[order[train_end:validation_end]] = 1

    network = _network_arrays(net)
    train_common = common_array[split == 0]
    lower = train_common.min(axis=0)
    upper = train_common.max(axis=0)
    lower[:, 2] = net.bus.min_vm_pu.to_numpy(dtype=np.float64)
    upper[:, 2] = net.bus.max_vm_pu.to_numpy(dtype=np.float64)
    build_seconds = time.perf_counter() - build_started
    metadata = {
        **asdict(config),
        "case_name": str(net.name),
        "base_mva": float(net.sn_mva),
        "num_buses": len(net.bus),
        "num_branches": int(len(network["branch_from"])),
        "state_common_channels": ["p_injection_pu", "q_injection_pu", "vm_pu", "theta_rad"],
        "state_wang_order": ["p_load_mw", "q_load_mvar", "vm_pu", "theta_rad", "p_gen_mw", "q_gen_mvar"],
        "attempts": attempts,
        "accepted": config.samples,
        "rejected_opf": failures,
        "opf_calls": opf_calls,
        "accepted_initialization_counts": initialization_counts,
        "build_seconds": build_seconds,
        "accepted_samples_per_second": config.samples / build_seconds,
        "cost_scale_columns": list(cost_columns),
        "split_codes": {"train": 0, "validation": 1, "test": 2},
        "provenance": {
            "load_range": "source-stated for wang/hoseinpour; common is project protocol",
            "split": "inferred because neither paper specifies a reusable split",
            "ieee118_sample_count": "project setting; not stated by Wang et al.",
            "solver": "AC-OPF, aligned with both source papers",
            "opf_initialization": (
                "PF warm start followed by flat retry"
                if config.retry_flat
                else "PF warm start without retry"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state_common=common_array,
        state_wang=wang_array,
        split=split,
        load_scale=load_scale_array,
        system_load_scale=system_scale_array,
        rejected_system_load_scale=rejected_system_scale_array,
        cost_scale=cost_scale_array,
        load_bus_indices=load_buses,
        generator_bus_indices=generator_buses,
        common_lower=lower.astype(np.float32),
        common_upper=upper.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        **network,
    )
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(output), "sha256": checksum, **metadata}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASE_FACTORIES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--protocol", choices=("common", "wang", "hoseinpour"), default="common")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--retry-flat",
        action="store_true",
        help="Retry a failed PF-warm-start OPF from a flat start. Disabled by default because the IEEE 30 diagnostic recovered 0/693 failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        args.output,
        BuildConfig(
            case=args.case,
            samples=args.samples,
            protocol=args.protocol,
            seed=args.seed,
            retry_flat=args.retry_flat,
        ),
    )


if __name__ == "__main__":
    main()
