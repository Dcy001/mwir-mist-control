"""The RT-PCM forward model for an explicitly supplied radiometric cycle."""

from dataclasses import dataclass

import numpy as np

from .physics_mist import (NozzleCalibration, RegionCalibration, SpraySystem,
                           droplet_number_density, effective_radius_m, liquid_water_content,
                           mist_transfer)
from .physics_spectral import (Atmosphere, SpectralGrid, _array, _positive, classify_los,
                               kruse_atmosphere, planck_um, plume_radiance, sea_radiance,
                               surface_radiance)


@dataclass(frozen=True)
class Plume:
    """World-ray entry/exit maps and far-to-near layers for an equivalent box (B.2)."""

    entry_m: np.ndarray
    exit_m: np.ndarray
    temperatures_k: np.ndarray
    absorption_m_inv: np.ndarray
    layer_fractions: np.ndarray


@dataclass(frozen=True)
class Boundary:
    """Observed/calibrated geometry and source terms for one cycle (Appendices A–D)."""

    camera_origin_m: np.ndarray
    rays: np.ndarray
    surface_distance_m: np.ndarray
    surface_temperature_k: np.ndarray
    surface_emissivity: np.ndarray
    sea_mask: np.ndarray
    sea_channel_means: np.ndarray
    sea_glint_weight: np.ndarray
    sea_reflectance: np.ndarray
    sea_emissivity: np.ndarray
    sea_temperature_k: np.ndarray
    solar_irradiance: np.ndarray
    atmosphere: Atmosphere
    plumes: tuple = ()


@dataclass(frozen=True)
class PhysicalResult:
    """Band-integrated frames [2,H,W] (W m^-2 sr^-1) and geometry diagnostics."""

    frames: np.ndarray
    fog_region: np.ndarray
    fog_chord_m: np.ndarray
    region_names: tuple
    atmosphere_source: str
    background_negative_samples: int


def _broadcast(name, values, shape, *, lower=None, upper=None):
    try:
        view = np.broadcast_to(np.asarray(values, dtype=float), shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    _array(name, np.asarray(values, dtype=float), lower=lower, upper=upper)
    return view


def _emissivity(name, values, shape, mask=None):
    values = np.asarray(values, dtype=float)
    if values.shape == shape[:-1]:
        values = values[..., None]
    try:
        view = np.broadcast_to(values, shape)
    except ValueError as exc:
        raise ValueError(f"{name} must be grey per pixel or spectral, broadcastable to {shape}") from exc
    return _array(name, view[mask] if mask is not None else view, lower=0, upper=1)


class RTPCM:
    """Prepare immutable-cycle sources once, then reevaluate mist for each action."""

    def __init__(self, spectral: SpectralGrid, boundary: Boundary, registry, spray: SpraySystem = None):
        self.spectral = spectral
        self.registry = registry
        self.spray = spray
        self.origin = _array("camera_origin_m", boundary.camera_origin_m)
        self.rays = _array("rays", boundary.rays)
        if self.origin.shape != (3,) or self.rays.ndim != 3 or self.rays.shape[-1] != 3 or not np.allclose(
            np.linalg.norm(self.rays, axis=-1), 1, atol=1e-7
        ):
            raise ValueError("Expected camera_origin_m [3] and unit rays [H,W,3]")
        self.shape = self.rays.shape[:2]
        self.region_names = tuple(dict.fromkeys(registry.regions))
        self.region_indices = [np.flatnonzero(np.asarray(registry.regions) == name) for name in self.region_names]
        self._geometry_cache = {}
        self._prepare(boundary)

    def _prepare(self, boundary):
        spectral_shape = (*self.shape, self.spectral.wavelength_um.size)
        wavelength = self.spectral.wavelength_um
        surface_distance = np.asarray(boundary.surface_distance_m, dtype=float)
        if surface_distance.shape != self.shape:
            raise ValueError("surface_distance_m must have shape [H,W]")
        nearest_entry = np.full(self.shape, np.inf)
        nearest_exit = np.full(self.shape, np.inf)
        plume_index = np.full(self.shape, -1, dtype=int)
        for index, plume in enumerate(boundary.plumes):
            entry = np.asarray(plume.entry_m, dtype=float)
            exit = np.asarray(plume.exit_m, dtype=float)
            if entry.shape != self.shape or exit.shape != self.shape:
                raise ValueError("Plume entry_m and exit_m must both have shape [H,W]")
            hit = np.isfinite(entry)
            invalid = (np.isnan(entry) | np.isnan(exit) | (entry < 0) | (exit < 0)
                       | (hit != np.isfinite(exit)) | (hit & (exit <= entry)))
            if np.any(invalid):
                raise ValueError("Each plume interval must be forward with positive length, or (+inf,+inf)")
            nearest = entry < nearest_entry
            nearest_entry[nearest], nearest_exit[nearest], plume_index[nearest] = entry[nearest], exit[nearest], index

        surface_mask, plume_mask, _ = classify_los(surface_distance, nearest_entry)
        self.foreground_m = np.minimum(surface_distance, nearest_entry)
        bg = self.spectral.background(boundary.sea_channel_means)
        self._negative_background_samples = int(np.count_nonzero(bg < 0))
        self.los = np.broadcast_to(bg, spectral_shape).copy()
        surface_temperature = np.asarray(boundary.surface_temperature_k, dtype=float)
        if surface_temperature.shape != self.shape:
            raise ValueError("surface_temperature_k must have shape [H,W]")
        if np.any(surface_mask):
            self.los[surface_mask] = surface_radiance(
                wavelength, surface_temperature[surface_mask],
                _emissivity("surface_emissivity", boundary.surface_emissivity, spectral_shape, surface_mask), bg
            )
        for index, plume in enumerate(boundary.plumes):
            belongs = plume_mask & (plume_index == index)
            fractions = _positive("plume.layer_fractions", plume.layer_fractions)
            if fractions.shape != np.shape(plume.temperatures_k) or not np.isclose(fractions.sum(), 1):
                raise ValueError("plume.layer_fractions must match temperatures_k and sum to 1")
            if np.any(belongs):
                chord = nearest_exit[belongs] - nearest_entry[belongs]
                self.los[belongs] = plume_radiance(wavelength, plume.temperatures_k,
                                                  plume.absorption_m_inv, chord[:, None] * fractions)

        self.sea_mask = np.asarray(boundary.sea_mask)
        if self.sea_mask.shape != self.shape or self.sea_mask.dtype != bool or not self.sea_mask.any():
            raise ValueError("sea_mask must be a nonempty Boolean mask [H,W]")
        sea = self.sea_mask
        sea_temperature = _broadcast("sea_temperature_k", boundary.sea_temperature_k, self.shape)[sea]
        sea_eps = _emissivity("sea_emissivity", boundary.sea_emissivity, spectral_shape, sea)
        sea_glint = _broadcast("sea_glint_weight", boundary.sea_glint_weight, self.shape)[sea]
        sea_reflectance = _broadcast("sea_reflectance", boundary.sea_reflectance, self.shape)[sea]
        self._no_mist_path = self.los.copy()
        self._no_mist_path[sea] += sea_radiance(wavelength, sea_temperature, sea_eps, sea_reflectance, sea_glint, 0)
        self._sea_solar_factor = np.zeros(self.shape)
        self._sea_solar_factor[sea] = sea_glint * sea_reflectance / np.pi
        self._solar = self.spectral.equivalent_solar(boundary.solar_irradiance)
        self._atm_source = boundary.atmosphere.source
        self._atm_t = _broadcast("atmosphere.transmittance", boundary.atmosphere.transmittance,
                                 spectral_shape, lower=0, upper=1)
        path = _broadcast("atmosphere.path_radiance", boundary.atmosphere.path_radiance, spectral_shape, lower=0)
        self._no_mist_frames = np.stack([
            self.spectral.integrate(self._atm_t * (self._no_mist_path + self._solar[c] * self._sea_solar_factor[..., None])
                                    + path, c)
            for c in range(2)
        ])

    def _curtain_geometry(self, action, index):
        indices = self.region_indices[index]
        key = (action.values[indices].tobytes(), action.enabled[indices].tobytes())
        cached = self._geometry_cache.get(index)
        if cached is not None and cached[0] == key:
            return cached[1]
        name = self.region_names[index]
        curtain, relative_speed = self.spray.curtain(self.registry, action, name, indices)
        entry, exit = curtain.intersect(self.origin, self.rays)
        hits = np.flatnonzero(np.isfinite(entry.ravel()))
        result = (curtain, relative_speed, hits, entry.ravel()[hits], exit.ravel()[hits])
        # Retain one sparse geometry per region within this boundary snapshot.
        self._geometry_cache[index] = (key, result)
        return result

    def evaluate(self, action):
        """Run C.1–C.19 and Eqs. (8)–(17) for the supplied candidate or actual setting."""
        flows = self.registry.flows(action)
        active = [float(flows[indices].sum()) for indices in self.region_indices]
        fog_region = np.full(self.foreground_m.size, -1, dtype=int)
        fog_entry = np.full(self.foreground_m.size, np.inf)
        fog_chord = np.zeros(self.foreground_m.size)
        curtains = {}
        frames = self._no_mist_frames.copy()
        if any(active) and self.spray is None:
            raise ValueError("Enabled spray requires spray geometry and optical calibration")
        for index, flow in enumerate(active):
            if flow == 0:
                continue
            curtain, relative_speed, hits, entry, exit = self._curtain_geometry(action, index)
            nearest = entry < fog_entry[hits]
            chosen = hits[nearest]
            fog_entry[chosen], fog_region[chosen] = entry[nearest], index
            fog_chord[chosen] = exit[nearest] - entry[nearest]
            curtains[index] = (curtain, relative_speed, hits)

        # C.12 discards a completely screened curtain. A partial overlap retains its chord.
        occluded = self.foreground_m.ravel() < fog_entry
        fog_region[occluded], fog_chord[occluded] = -1, 0
        for index, (curtain, relative_speed, hits) in curtains.items():
            selected = hits[fog_region[hits] == index]
            if not selected.size:
                continue
            pixels = np.unravel_index(selected, self.shape)
            calibration = curtain.calibration
            radius = effective_radius_m(calibration.diameter50_m, calibration.sigma_g)
            water_density = self.spray.water_density_kg_m3
            lwc = liquid_water_content(active[index], relative_speed, calibration.area_m2(), water_density)
            density = droplet_number_density(lwc, radius, water_density)
            cosine = np.clip(-self.rays[pixels] @ curtain.axis, -1, 1)
            fog_t, fog_correction, _ = mist_transfer(
                self.spectral.wavelength_um, self.spray.water_nk, density, radius, fog_chord[selected],
                cosine, self.los[pixels],
            )
            # Subtract the no-mist path from the mist path before integrating. Atmospheric
            # self-emission cancels in this difference and is already in _no_mist_frames.
            atmospheric_t = self._atm_t[pixels]
            sea_factor = self._sea_solar_factor[pixels][:, None]
            for c in range(2):
                reference = self._no_mist_path[pixels] + self._solar[c] * sea_factor
                delta = (fog_t - 1) * reference + fog_correction
                frames[c][pixels] += self.spectral.integrate(atmospheric_t * delta, c)
        return PhysicalResult(frames, fog_region.reshape(self.shape), fog_chord.reshape(self.shape),
                              self.region_names, self._atm_source, self._negative_background_samples)

