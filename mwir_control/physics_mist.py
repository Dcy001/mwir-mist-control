"""Appendix C spray geometry, liquid loading and Mie optical properties."""

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Mapping

import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import spherical_jn, spherical_yn
from scipy.spatial.transform import Rotation

from .geometry import ray_box_interval
from .physics_spectral import _array, _positive


SC_NOMINAL_FAN_WIDTH_M = 1.27
SC_FAN_REFERENCE_DISTANCE_M = 0.508


def _vector(name, values):
    vector = _array(name, values)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape [3]")
    return vector


def _unit(name, values):
    vector = _vector(name, values)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError(f"{name} cannot define a direction")
    return vector / length


@dataclass(frozen=True)
class NozzleCalibration:
    """Tabulated per-pressure geometry at the supplied inflow condition."""

    pressure_mpa: np.ndarray
    jet_speed_mps: np.ndarray
    curvature_world_m_inv: np.ndarray
    length_m: np.ndarray

    def __post_init__(self):
        pressure = _positive("calibration.pressure_mpa", self.pressure_mpa)
        speed = _positive("calibration.jet_speed_mps", self.jet_speed_mps)
        curvature = _array("calibration.curvature_world_m_inv", self.curvature_world_m_inv)
        length = _positive("calibration.length_m", self.length_m)
        if (pressure.ndim != 1 or pressure.size < 2 or np.any(np.diff(pressure) <= 0)
                or speed.shape != pressure.shape or length.shape != pressure.shape
                or curvature.shape != (pressure.size, 3)):
            raise ValueError("Nozzle calibration requires sorted pressure points, speed/length [P] and curvature [P,3]")
        for name, value in (("pressure_mpa", pressure), ("jet_speed_mps", speed),
                            ("curvature_world_m_inv", curvature), ("length_m", length)):
            object.__setattr__(self, name, value)

    def at(self, pressure):
        """Return speed, world curvature and effective length at a calibrated pressure."""
        grid = self.pressure_mpa
        if not grid[0] <= pressure <= grid[-1]:
            raise ValueError(f"pressure_mpa {pressure:g} is outside nozzle geometry calibration [{grid[0]:g}, {grid[-1]:g}]")
        curvature = np.array([np.interp(pressure, grid, row) for row in self.curvature_world_m_inv.T])
        return np.interp(pressure, grid, self.jet_speed_mps), curvature, np.interp(pressure, grid, self.length_m)


@dataclass(frozen=True)
class RegionCalibration:
    """Parameters of Eqs. (C.4)-(C.5), (C.14)-(C.16). Fan dimensions select SC."""

    length_m: float
    initial_half_thickness_m: float
    growth: float
    diameter50_m: float
    sigma_g: float
    fan_width_m: float = None
    fan_ref_distance_m: float = None

    def __post_init__(self):
        _positive("region length, initial thickness and median diameter",
                  (self.length_m, self.initial_half_thickness_m, self.diameter50_m))
        _array("region growth", self.growth, lower=0)
        _array("region sigma_g", self.sigma_g, lower=1)
        if self.fan_width_m is not None or self.fan_ref_distance_m is not None:
            _positive("region fan width and reference distance", (self.fan_width_m, self.fan_ref_distance_m))

    def sections(self):
        """Intervals in s/length and polynomial coefficients for width and thickness."""
        end = self.length_m
        thickness = Polynomial([self.initial_half_thickness_m, self.growth * end])
        if self.fan_width_m is None:
            return [(0.0, 1.0, thickness, thickness)]
        breakpoint = min(self.fan_ref_distance_m / end, 1.0)
        sections = [(0.0, breakpoint, Polynomial([0, self.fan_width_m * end / (2 * self.fan_ref_distance_m)]), thickness)]
        if breakpoint < 1:
            sections.append((breakpoint, 1.0, Polynomial([self.fan_width_m / 2]), thickness))
        return sections

    def area_m2(self):
        area = 0.0
        for start, stop, width, thickness in self.sections():
            primitive = (width * thickness).integ()
            area += np.pi * (primitive(stop) - primitive(start))
        return area


def effective_radius_m(diameter50_m, sigma_g):
    """Eq. (C.15), analytic lognormal third-to-second moment ratio."""
    median = _positive("diameter50_m", diameter50_m) / 2
    sigma = _array("sigma_g", sigma_g, lower=1)
    return median * np.exp(2.5 * np.log(sigma)**2)


def liquid_water_content(flow_lpm, relative_speed_mps, cross_section_m2, water_density_kg_m3):
    """Eq. (C.14) in kg m^-3, including litres/minute to m^3/second."""
    flow = _array("flow_lpm", flow_lpm, lower=0)
    speed = _positive("relative_speed_mps", relative_speed_mps)
    area = _positive("cross_section_m2", cross_section_m2)
    density = _positive("water_density_kg_m3", water_density_kg_m3)
    return density * flow / (60000 * speed * area)


def droplet_number_density(lwc_kg_m3, radius_m, water_density_kg_m3):
    """Eq. (C.16) in droplets m^-3."""
    lwc = _array("lwc_kg_m3", lwc_kg_m3, lower=0)
    density = _positive("water_density_kg_m3", water_density_kg_m3)
    radius = _positive("radius_m", radius_m)
    return lwc / (4 / 3 * np.pi * density * radius**3)


@lru_cache(maxsize=16384)
def _mie_sphere(size, refractive_index):
    """Mie efficiencies and scattering asymmetry for an absorbing sphere."""
    if size < 1e-3:
        polarizability = (refractive_index**2 - 1) / (refractive_index**2 + 2)
        scattering = 8 / 3 * size**4 * abs(polarizability)**2
        extinction = 4 * size * polarizability.imag + scattering
        return extinction, scattering, 0.0

    count = int(np.ceil(size + 4.05 * size**(1 / 3) + 2))
    n = np.arange(1, count + 1)
    interior = refractive_index * size
    high_order = max(count, int(abs(interior))) + 25
    logarithmic_derivative = np.zeros(high_order + 1, dtype=complex)
    for order in range(high_order, 0, -1):
        term = order / interior
        logarithmic_derivative[order - 1] = term - 1 / (logarithmic_derivative[order] + term)
    derivative = logarithmic_derivative[n]

    orders = np.arange(count + 1)
    psi = size * spherical_jn(orders, size)
    xi = psi + 1j * size * spherical_yn(orders, size)
    a_term = derivative / refractive_index + n / size
    b_term = refractive_index * derivative + n / size
    a = (a_term * psi[1:] - psi[:-1]) / (a_term * xi[1:] - xi[:-1])
    b = (b_term * psi[1:] - psi[:-1]) / (b_term * xi[1:] - xi[:-1])
    extinction = 2 / size**2 * np.sum((2 * n + 1) * (a + b).real)
    scattering = 2 / size**2 * np.sum((2 * n + 1) * (abs(a)**2 + abs(b)**2))
    asymmetry = 0.0
    if scattering > 0:
        adjacent = n[:-1] * (n[:-1] + 2) / (n[:-1] + 1)
        same_order = (2 * n + 1) / (n * (n + 1))
        g_sum = (np.sum(adjacent * (a[:-1] * a[1:].conj() + b[:-1] * b[1:].conj()).real)
                 + np.sum(same_order * (a * b.conj()).real))
        asymmetry = 4 * g_sum / (size**2 * scattering)
    return extinction, scattering, asymmetry


def mie_optics(wavelength_um, water_nk, radius_m):
    """Return Q_ext, single-scattering albedo and asymmetry at the effective radius."""
    wavelength = _positive("wavelength_um", wavelength_um)
    nk = np.asarray(water_nk, dtype=complex)
    radius = float(_positive("radius_m", radius_m))
    if nk.shape != wavelength.shape or not np.all(np.isfinite(nk) & (nk.real > 0) & (nk.imag >= 0)):
        raise ValueError("water_nk needs one finite n+i*k per wavelength with n>0 and k>=0")
    size = 2 * np.pi * radius / (wavelength * 1e-6)
    extinction, scattering, asymmetry = np.array([
        _mie_sphere(float(x), complex(m)) for x, m in zip(size, nk)
    ]).T
    if not (np.isfinite(extinction).all() and np.isfinite(scattering).all() and np.isfinite(asymmetry).all()):
        raise ValueError("Mie optical calculation returned non-finite coefficients")
    if np.any((extinction < 0) | (scattering < 0) | (scattering > extinction * (1 + 1e-8)) | (abs(asymmetry) >= 1)):
        raise ValueError("Mie optical calculation failed its physical-domain checks")
    omega = np.divide(scattering, extinction, out=np.zeros_like(scattering), where=extinction > 0)
    return extinction, np.minimum(omega, 1), asymmetry


def orientation_weight(asymmetry, direction_cosine):
    """Eqs. (C.18)-(C.19), evaluating the printed expression."""
    g = _array("asymmetry", asymmetry)
    cosine = _array("direction_cosine", direction_cosine, lower=-1, upper=1)
    if np.any(abs(g) >= 1):
        raise ValueError("asymmetry requires -1 < g < 1")
    return ((1 - g**2) / np.sqrt(4 * np.pi * ((1 - abs(g))**2 + 2 * abs(g) * (1 - np.sign(g) * cosine))))**3


@dataclass(frozen=True)
class Curtain:
    """Swept transverse ellipses of a calibrated region, in world metres."""

    centre_coefficients: np.ndarray
    axis: np.ndarray
    width_axis: np.ndarray
    thickness_axis: np.ndarray
    calibration: RegionCalibration

    def _bounds(self):
        centre = self.centre_coefficients
        # A conservative box encloses the quadratic using its Bézier control points.
        control = np.array([centre[0], centre[0] + centre[1] / 2, centre.sum(axis=0)])
        last = self.calibration.sections()[-1]
        margin = abs(self.width_axis) * last[2](1) + abs(self.thickness_axis) * last[3](1)
        return control.min(axis=0) - margin - 1e-9, control.max(axis=0) + margin + 1e-9

    def intersect(self, origin, rays):
        """Eqs. (C.6)-(C.11). Return min and max forward LOS distances."""
        centre = self.centre_coefficients
        longitudinal = Polynomial((centre - np.array([origin, [0, 0, 0], [0, 0, 0]])) @ self.axis)
        if min(longitudinal.deriv()(0.0), longitudinal.deriv()(1.0)) <= 0:
            raise ValueError("Calibrated curtain centreline folds back along its regional axis")
        widths = self.calibration.sections()
        entry, exit = ray_box_interval(origin, rays, *self._bounds())
        possible = np.flatnonzero(np.isfinite(entry.ravel()))
        ray_rows = rays.reshape(-1, 3)
        entry, exit = np.full(entry.size, np.inf), np.full(exit.size, np.inf)
        for index in possible:
            direction = ray_rows[index]
            axial = direction @ self.axis
            if abs(axial) < 1e-12:
                lo, hi = self._transverse_intersection(origin, direction, longitudinal, widths)
            else:
                distance = longitudinal / axial
                offsets = np.array([origin, [0, 0, 0], [0, 0, 0]]) - centre
                w = Polynomial(offsets @ self.width_axis) + distance * (direction @ self.width_axis)
                t = Polynomial(offsets @ self.thickness_axis) + distance * (direction @ self.thickness_axis)
                intervals = []
                for start, stop, b, h in widths:
                    edge = (w * h)**2 + (t * b)**2 - (b * h)**2
                    points = [start, *_real_roots(edge, start, stop), stop]
                    for a, z in zip(points, points[1:]):
                        if z > a and edge((a + z) / 2) <= 0:
                            intervals.append(sorted((distance(a), distance(z))))
                lo = min((max(0, a) for a, z in intervals if z > 0), default=np.inf)
                hi = max((z for a, z in intervals if z > 0), default=np.inf)
            if hi > lo:
                entry[index], exit[index] = lo, hi
        return entry.reshape(rays.shape[:-1]), exit.reshape(rays.shape[:-1])

    def _transverse_intersection(self, origin, ray, longitudinal, sections):
        endpoints = [x for x in (0.0, 1.0) if abs(longitudinal(x)) < 1e-12]
        roots = endpoints + _real_roots(longitudinal, 0.0, 1.0)
        if not roots:
            return np.inf, np.inf
        u = roots[0]
        start, stop, b, h = next(piece for piece in sections if piece[0] <= u <= piece[1])
        if b(u) <= 0:
            return np.inf, np.inf
        offset = origin - np.array([1, u, u*u]) @ self.centre_coefficients
        cross = np.array([self.width_axis / b(u), self.thickness_axis / h(u)])
        c, d = cross @ offset, cross @ ray
        coefficients = [c @ c - 1, 2 * c @ d, d @ d]
        roots = np.sort(Polynomial(coefficients).roots())
        if len(roots) != 2 or np.iscomplexobj(roots) or roots[-1] <= 0:
            return np.inf, np.inf
        return max(0, roots[0]), roots[1]


def _real_roots(polynomial, start, stop):
    roots = polynomial.roots()
    return sorted(float(x.real) for x in roots if abs(x.imag) < 1e-9 and start < x.real < stop)


@dataclass(frozen=True)
class SpraySystem:
    """Snapshot of geometry, weather and externally identified spray calibration."""

    ship_to_world_rotation: np.ndarray
    ship_origin_m: np.ndarray
    inflow_wind_world_mps: np.ndarray
    water_nk: np.ndarray
    water_density_kg_m3: float
    regions: Mapping
    nozzles: Mapping

    def __post_init__(self):
        rotation = _array("ship_to_world_rotation", self.ship_to_world_rotation)
        if (rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
            raise ValueError("ship_to_world_rotation must be a proper rotation")
        object.__setattr__(self, "ship_to_world_rotation", rotation)
        object.__setattr__(self, "ship_origin_m", _vector("ship_origin_m", self.ship_origin_m))
        object.__setattr__(self, "inflow_wind_world_mps", _vector("inflow_wind_world_mps", self.inflow_wind_world_mps))
        _positive("water_density_kg_m3", self.water_density_kg_m3)

    def curtain(self, registry, action, name, indices):
        """Eq. (C.1)-(C.5), with candidate-dependent orientation, calibration and wind."""
        try:
            region = self.regions[name]
        except KeyError as exc:
            raise ValueError(f"Missing spray.regions calibration for {name}") from exc
        actions = action.values[indices]
        if not action.enabled[indices].all():
            raise ValueError(f"{name}: mixed enabled/disabled group requires a separate envelope calibration")
        if callable(region):
            region = region(actions, self.inflow_wind_world_mps)
        is_stack = name in ("SCC-F", "SCC-A")
        if is_stack and region.fan_width_m is not None:
            raise ValueError(f"{name}: an SCC envelope does not use flat-fan dimensions")
        if not is_stack and region.fan_width_m is None:
            region = replace(region, fan_width_m=SC_NOMINAL_FAN_WIDTH_M,
                             fan_ref_distance_m=SC_FAN_REFERENCE_DISTANCE_M)
        centres, axes, widths, jet_velocity = [], [], [], []
        for index, values in zip(indices, actions):
            nozzle = registry.nozzles[index]
            pressure, _, deflection_deg = values
            try:
                calibration = self.nozzles[nozzle.nozzle_id]
            except KeyError as exc:
                raise ValueError(f"Missing spray.nozzles calibration for {nozzle.nozzle_id}") from exc
            try:
                if callable(calibration):
                    speed, curvature, length = calibration(values, self.inflow_wind_world_mps)
                else:
                    speed, curvature, length = calibration.at(pressure)
                speed = float(_positive("calibrated jet speed", speed))
                length = float(_positive("calibrated nozzle length", length))
                curvature = _vector("calibrated nozzle curvature", curvature)
                if region.length_m > length:
                    raise ValueError("region length exceeds calibrated nozzle centreline length")
            except ValueError as exc:
                raise ValueError(f"{nozzle.nozzle_id}: {exc}") from exc
            deflection = Rotation.from_rotvec(np.deg2rad(deflection_deg) * nozzle.deflection_axis).as_matrix()
            frame = self.ship_to_world_rotation @ nozzle.mount_rotation @ deflection
            anchor = self.ship_origin_m + self.ship_to_world_rotation @ nozzle.position_m
            centre = np.array([anchor, frame[:, 2] * region.length_m, curvature * (region.length_m**2 / 2)])
            centres.append(centre)
            axes.append(frame[:, 2])
            widths.append(frame[:, 0])
            jet_velocity.append(speed * frame[:, 2])
        axis = _unit("regional spray axis", np.sum(axes, axis=0))
        if len(indices) == 1:
            width = widths[0]
        else:
            reference = np.eye(3)[np.argmin(abs(axis))]
            width = _unit("regional width", np.cross(axis, reference))
        curtain = Curtain(np.mean(centres, axis=0), axis, width, np.cross(axis, width), region)
        relative_speed = np.linalg.norm(np.mean(jet_velocity, axis=0) - self.inflow_wind_world_mps)
        return curtain, relative_speed


def mist_transfer(wavelength_um, spectral_index, density, radius, chord_m, direction_cosine, los_radiance):
    """Eqs. (8)-(11). Return mist transmittance, path correction and orientation."""
    q_ext, omega, g = mie_optics(wavelength_um, spectral_index, radius)
    depth = _array("chord_m", chord_m, lower=0)[..., None] * q_ext * np.pi * radius**2 * density
    orientation = orientation_weight(g, np.asarray(direction_cosine)[..., None])
    transmittance = np.exp(-depth)
    correction = omega * orientation * -np.expm1(-depth) * los_radiance
    return transmittance, correction, orientation

