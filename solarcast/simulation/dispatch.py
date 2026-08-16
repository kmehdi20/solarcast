"""Battery dispatch strategies.

Each strategy is a pure function: it takes the instantaneous state (PV,
load, battery SOC) and returns the **requested** battery power (positive
= charge, negative = discharge), *before* physical limits are applied —
those limits are applied afterward by `Battery.charge()` /
`Battery.discharge()` in the engine (`engine.py`).

This separation lets a dispatch policy be tested independently of the
battery's physical constraints, and lets strategies be compared on the
same dataset without duplicating the simulation engine.

Design note
------------
Both strategies provided here always return a requested power whose sign
matches the PV-load net (`pv_kw - load_kw`): they never charge from the
grid and never discharge to the grid. A forecast-driven strategy
(pre-charging overnight ahead of an announced cloudy day, for example)
could lift that constraint; the engine handles that case correctly (see
the derivation in `engine.py`), but no such strategy is provided here yet.
"""

from __future__ import annotations

from typing import Protocol


class DispatchStrategy(Protocol):
    def __call__(self, pv_kw: float, load_kw: float, soc_fraction: float) -> float: ...


def self_consumption(pv_kw: float, load_kw: float, soc_fraction: float) -> float:
    """Maximize self-consumption: charge on PV surplus, discharge on deficit.

    The reference strategy, the simplest and most common choice for a
    residential installation with no variable tariff or forecast.
    """
    return pv_kw - load_kw


def peak_shaving(
    pv_kw: float,
    load_kw: float,
    soc_fraction: float,
    import_threshold_kw: float = 3.0,
    export_threshold_kw: float = 3.0,
) -> float:
    """Self-consumption with clipping: only reacts beyond a threshold.

    Useful when the grid contract bills peak demanded power: the battery
    only intervenes to bring import or export back under the set
    threshold, and lets the rest flow through the grid — preserving SOC
    for larger excursions instead of spending it on small deviations.
    """
    net = pv_kw - load_kw
    if net > export_threshold_kw:
        return net - export_threshold_kw
    if net < -import_threshold_kw:
        return net + import_threshold_kw
    return 0.0


STRATEGIES: dict[str, DispatchStrategy] = {
    "self_consumption": self_consumption,
    "peak_shaving": peak_shaving,
}


def get_strategy(name: str) -> DispatchStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name]
