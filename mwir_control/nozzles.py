"""Nozzle identity, working-domain projection and duty-averaged hydraulics."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


SC_REGION_IDS = tuple(f"SC_{i}" for i in range(1, 279))
SCC_PRESSURES = np.array([0.20, 0.40, 0.60, 0.80, 0.827])
SCC_A_TOTAL = np.array([644.6, 911.6, 1116.6, 1289.5, 1311.0])
SCC_F_TOTAL = np.array([1106.8, 1565.2, 1917.2, 2214.1, 2251.0])
SC_PRESSURES = np.array([0.20, 0.52, 0.62, 0.69, 0.83, 0.90])
SC_FLOWS = np.array([2.50, 3.95, 4.32, 4.69, 5.00, 5.30])
WORKING_BOUNDS = {
    "SCC": ((0.727, 0.60, -12.0), (0.927, 1.0, 12.0)),
    "SC": ((0.137, 0.40, -8.0), (0.357, 1.0, 8.0)),
}


def _vector(value, shape, field_name):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


def published_flow_curve(kind):
    """Return pressure (MPa) and single-nozzle flow at unit duty (L/min).

    The SCC-A and SCC-F layouts in Fig. 6 and Table F.1 yield two
    equations with Port/Starboard sharing one curve (Appendix C.5).
    """
    if kind == "SC":
        return SC_PRESSURES.copy(), SC_FLOWS.copy()
    q_ps = (SCC_F_TOTAL - SCC_A_TOTAL) / 2
    if kind in ("SCC-port", "SCC-starboard"):
        return SCC_PRESSURES.copy(), q_ps
    if kind == "SCC-midships":
        return SCC_PRESSURES.copy(), SCC_A_TOTAL - 2 * q_ps
    raise ValueError(f"Unknown nozzle kind {kind!r}")


@dataclass(frozen=True)
class Action:
    """One registry-ordered setting per nozzle, [MPa, fraction, deg]."""

    values: np.ndarray
    enabled: np.ndarray

    def __post_init__(self):
        enabled = np.array(self.enabled, copy=True)
        if enabled.ndim != 1 or enabled.dtype != np.dtype(bool):
            raise ValueError("enabled must be a one-dimensional Boolean array")
        values = _vector(self.values, (enabled.size, 3), "action values")
        pressure, duty, _ = values.T
        if (pressure < 0).any() or (duty < 0).any() or (duty > 1).any():
            raise ValueError("Action pressure must be non-negative and duty in [0,1]")
        if np.any((pressure[enabled] <= 0) | (duty[enabled] <= 0)):
            raise ValueError("Enabled nozzles require positive pressure and duty")
        if np.any(duty[~enabled] != 0):
            raise ValueError("Disabled nozzles require zero duty")
        enabled.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "enabled", enabled)

    @classmethod
    def off(cls, size):
        return cls(np.zeros((size, 3)), np.zeros(size, dtype=bool))


@dataclass(frozen=True)
class Nozzle:
    nozzle_id: str
    region: str
    kind: str
    position_m: np.ndarray
    mount_rotation: np.ndarray
    deflection_axis: np.ndarray
    reference_pressure_mpa: float
    flow_extension: tuple = ()
    _pressure: np.ndarray = field(init=False, repr=False)
    _flow: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        name = self.nozzle_id
        if not isinstance(name, str) or not name.strip():
            raise ValueError("nozzle_id must be a non-empty string")
        if not isinstance(self.region, str) or not self.region.strip() or self.region in ("Sea", "Global"):
            raise ValueError(f"{name}: invalid spray region {self.region!r}")
        pressure, flow = published_flow_curve(self.kind)
        stack_region = self.region in ("SCC-F", "SCC-A")
        if (self.kind != "SC") != stack_region:
            raise ValueError(f"{name}: kind {self.kind} does not belong to {self.region}")
        position = _vector(self.position_m, (3,), f"{name}.position_m")
        rotation = _vector(self.mount_rotation, (3, 3), f"{name}.mount_rotation")
        axis = _vector(self.deflection_axis, (3,), f"{name}.deflection_axis")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1, atol=1e-6
        ):
            raise ValueError(f"{name}.mount_rotation must be a proper rotation")
        if not np.isclose(np.linalg.norm(axis), 1, atol=1e-6):
            raise ValueError(f"{name}.deflection_axis must be a unit vector in the mount frame")
        if len(self.flow_extension):
            extension = np.asarray(self.flow_extension, dtype=float)
            if extension.ndim != 2 or extension.shape[1] != 2:
                raise ValueError(f"{name}.flow_extension must contain [MPa, L/min] pairs")
            if not np.isfinite(extension).all() or np.any(extension <= 0):
                raise ValueError(f"{name}.flow_extension must be positive and finite")
            if np.any((extension[:, 0] >= pressure[0]) & (extension[:, 0] <= pressure[-1])):
                raise ValueError(f"{name}.flow_extension may only extend the published pressure range")
            pressure = np.concatenate((pressure, extension[:, 0]))
            flow = np.concatenate((flow, extension[:, 1]))
        order = np.argsort(pressure)
        pressure = pressure[order]
        flow = flow[order]
        if np.any(np.diff(pressure) <= 0):
            raise ValueError(f"{name}: duplicated flow calibration pressure")
        for key, value in (
            ("position_m", position), ("mount_rotation", rotation), ("deflection_axis", axis),
            ("_pressure", _vector(pressure, pressure.shape, f"{name}.pressure_calibration")),
            ("_flow", _vector(flow, flow.shape, f"{name}.flow_calibration")),
        ):
            object.__setattr__(self, key, value)
        reference = float(self.reference_pressure_mpa)
        object.__setattr__(self, "reference_pressure_mpa", reference)
        if not self.bounds[0][0] <= reference <= self.bounds[1][0]:
            raise ValueError(f"{name}.reference_pressure_mpa must be inside the working domain")
        self.flow_l_min(reference, 1.0)

    @property
    def axis(self):
        return self.mount_rotation[:, 2]

    @property
    def bounds(self):
        return WORKING_BOUNDS["SC" if self.kind == "SC" else "SCC"]

    def flow_l_min(self, pressure_mpa, duty):
        """Appendix C.13, piecewise linear between calibration points."""
        if not np.isfinite(pressure_mpa) or not self._pressure[0] <= pressure_mpa <= self._pressure[-1]:
            raise ValueError(
                f"{self.nozzle_id}: pressure {pressure_mpa:g} MPa outside flow calibration "
                f"[{self._pressure[0]:g}, {self._pressure[-1]:g}] MPa; supply flow_extension"
            )
        if not np.isfinite(duty) or not 0 <= duty <= 1:
            raise ValueError(f"{self.nozzle_id}: duty must be in [0,1]")
        return float(duty * np.interp(pressure_mpa, self._pressure, self._flow))


class NozzleRegistry:
    """Stable ordering for every nozzle-indexed array.

    Direct construction also supports reduced unit-test fixtures. Production
    input should use from_rows(), which validates the hardware inventory.
    """

    def __init__(self, nozzles):
        self.nozzles = tuple(nozzles)
        self.ids = tuple(nozzle.nozzle_id for nozzle in self.nozzles)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Registry must contain unique nozzle IDs")
        self.regions = tuple(nozzle.region for nozzle in self.nozzles)
        self.region_ids = tuple(dict.fromkeys(self.regions))
        self._stack_groups = {
            region: np.flatnonzero(np.asarray(self.regions) == region)
            for region in ("SCC-F", "SCC-A") if region in self.region_ids
        }
        self.lower = _vector([nozzle.bounds[0] for nozzle in self.nozzles], (len(self.ids), 3), "lower bounds")
        self.upper = _vector([nozzle.bounds[1] for nozzle in self.nozzles], (len(self.ids), 3), "upper bounds")

    def __len__(self):
        return len(self.nozzles)

    @classmethod
    def from_rows(cls, rows):
        required = (
            "nozzle_id", "region", "kind", "position_m", "mount_rotation",
            "deflection_axis", "reference_pressure_mpa",
        )
        nozzles = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"nozzles[{index}] must be a mapping")
            for key in required:
                if key not in row:
                    raise ValueError(f"nozzles[{index}].{key} is required")
            nozzles.append(Nozzle(**{key: row[key] for key in required}, flow_extension=row.get("flow_extension", ())))
        result = cls(nozzles)
        for region, per_side in (("SCC-F", 2), ("SCC-A", 1)):
            expected = {"SCC-port": per_side, "SCC-starboard": per_side, "SCC-midships": 1}
            if Counter(n.kind for n in nozzles if n.region == region) != expected:
                raise ValueError(f"{region} requires nozzle inventory {expected}")
        side_nozzles = [nozzle for nozzle in nozzles if nozzle.kind == "SC"]
        expected_regions = set(SC_REGION_IDS)
        if len(side_nozzles) != 278 or {nozzle.region for nozzle in side_nozzles} != expected_regions:
            raise ValueError("Registry requires 278 SC nozzles, one per ROI SC_1 through SC_278")
        if Counter(np.sign(nozzle.position_m[1]) for nozzle in side_nozzles) != {-1: 139, 1: 139}:
            raise ValueError("SC inventory requires 139 port (+Y) and 139 starboard (-Y) nozzles")
        return result

    def validate_enabled(self, enabled):
        enabled = np.asarray(enabled)
        if enabled.shape != (len(self),) or enabled.dtype.kind != "b":
            raise ValueError("enabled must be Boolean [N] in registry order")
        for region, indices in self._stack_groups.items():
            group = enabled[indices]
            if group.any() and not group.all():
                raise ValueError(f"{region} requires all nozzles enabled or all disabled")
        return enabled

    def project(self, values, enabled):
        """Eq. (22) for enabled nozzles, with a canonical zero row otherwise."""
        enabled = self.validate_enabled(enabled)
        if values is None:
            if enabled.any():
                raise ValueError("raw prediction is required for enabled nozzles")
            return Action.off(len(self))
        raw = np.asarray(values, dtype=float)
        if raw.shape != (len(self), 3) or not np.isfinite(raw[enabled]).all() or np.isinf(raw).any():
            raise ValueError("raw prediction must be [N,3] and finite for enabled nozzles")
        projected = np.zeros(raw.shape)
        projected[enabled] = np.clip(raw[enabled], self.lower[enabled], self.upper[enabled])
        return Action(projected, enabled)

    def validate_action(self, action, feasible=False):
        if not isinstance(action, Action) or action.values.shape != (len(self), 3):
            raise ValueError("Action shape does not match nozzle registry")
        self.validate_enabled(action.enabled)
        if feasible:
            enabled = action.enabled
            u = action.values[enabled]
            if np.any((u < self.lower[enabled]) | (u > self.upper[enabled])):
                raise ValueError("Enabled action is outside the Table F.1 working domain")

    def flows(self, action):
        self.validate_action(action)
        return np.array([
            nozzle.flow_l_min(values[0], values[1]) if enabled else 0.0
            for nozzle, values, enabled in zip(self.nozzles, action.values, action.enabled)
        ])

    def region_flows(self, action):
        total = dict.fromkeys(self.region_ids, 0.0)
        for region, rate in zip(self.regions, self.flows(action)):
            total[region] += float(rate)
        return total
