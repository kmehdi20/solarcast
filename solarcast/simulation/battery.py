"""Stationary battery model.

Fixed-timestep energy balance, with:

* charge/discharge power limits (inverter-battery);
* capacity limits (min/max SOC — e.g. 10%-95% to preserve lifespan, a
  common Li-ion practice);
* separate charge and discharge efficiencies. The usual round-trip
  efficiency for residential Li-ion (~90-95%) is split roughly evenly
  between the two directions; the defaults target ~92% round-trip.

Sign convention: in the dispatch engine's results (`engine.py`), positive
power = charging (energy entering the battery), negative = discharging.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatterySpec:
    """Technical characteristics of the battery."""

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float = 0.96
    discharge_efficiency: float = 0.96
    min_soc: float = 0.10
    max_soc: float = 0.95
    initial_soc: float = 0.50

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive.")
        if self.max_charge_kw < 0 or self.max_discharge_kw < 0:
            raise ValueError("max_charge_kw and max_discharge_kw must be zero or positive.")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("efficiencies must be within ]0, 1].")
        if not 0 <= self.min_soc < self.max_soc <= 1:
            raise ValueError("must have 0 <= min_soc < max_soc <= 1.")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("initial_soc must be between min_soc and max_soc.")


class Battery:
    """Battery with a mutable state of charge (SOC), driven step by step.

    Usage
    -----
    >>> battery = Battery(BatterySpec(capacity_kwh=10, max_charge_kw=3, max_discharge_kw=3))
    >>> accepted_kw = battery.charge(requested_kw=3.0, dt_hours=1.0)
    >>> delivered_kw = battery.discharge(requested_kw=2.0, dt_hours=1.0)
    """

    def __init__(self, spec: BatterySpec) -> None:
        self.spec = spec
        self.soc_kwh = spec.initial_soc * spec.capacity_kwh
        self.total_charged_kwh = 0.0
        self.total_discharged_kwh = 0.0

    @property
    def soc_fraction(self) -> float:
        return self.soc_kwh / self.spec.capacity_kwh

    @property
    def _min_kwh(self) -> float:
        return self.spec.min_soc * self.spec.capacity_kwh

    @property
    def _max_kwh(self) -> float:
        return self.spec.max_soc * self.spec.capacity_kwh

    def charge(self, requested_kw: float, dt_hours: float) -> float:
        """Attempt to absorb `requested_kw` of charging over `dt_hours`.

        Returns the power actually accepted on the external side (kW),
        bounded by `max_charge_kw` and by the remaining headroom in the
        battery given the charge efficiency (part of the absorbed energy
        is lost as heat, so less external power is accepted than the
        available headroom alone would suggest).
        """
        if requested_kw <= 0 or dt_hours <= 0:
            return 0.0

        power_kw = min(requested_kw, self.spec.max_charge_kw)

        headroom_kwh = self._max_kwh - self.soc_kwh
        max_storable_kw = headroom_kwh / (self.spec.charge_efficiency * dt_hours)
        power_kw = max(0.0, min(power_kw, max_storable_kw))

        stored_kwh = power_kw * dt_hours * self.spec.charge_efficiency
        self.soc_kwh = min(self._max_kwh, self.soc_kwh + stored_kwh)
        self.total_charged_kwh += stored_kwh
        return power_kw

    def discharge(self, requested_kw: float, dt_hours: float) -> float:
        """Attempt to deliver `requested_kw` of discharge over `dt_hours`.

        Returns the power actually delivered on the external side (kW),
        bounded by `max_discharge_kw` and by the energy available above
        `min_soc`, given the discharge efficiency (more internal energy
        must be drawn than what's actually delivered externally).
        """
        if requested_kw <= 0 or dt_hours <= 0:
            return 0.0

        power_kw = min(requested_kw, self.spec.max_discharge_kw)

        available_kwh = self.soc_kwh - self._min_kwh
        max_deliverable_kw = (available_kwh * self.spec.discharge_efficiency) / dt_hours
        power_kw = max(0.0, min(power_kw, max_deliverable_kw))

        drawn_kwh = power_kw * dt_hours / self.spec.discharge_efficiency
        self.soc_kwh = max(self._min_kwh, self.soc_kwh - drawn_kwh)
        self.total_discharged_kwh += power_kw * dt_hours
        return power_kw

    def reset(self) -> None:
        """Reset the battery to its initial state. Useful between simulations."""
        self.soc_kwh = self.spec.initial_soc * self.spec.capacity_kwh
        self.total_charged_kwh = 0.0
        self.total_discharged_kwh = 0.0
