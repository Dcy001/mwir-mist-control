"""Spectral radiometry in wavelength micrometres and kelvin (Appendices B and D)."""

from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann, Planck, speed_of_light
from scipy.interpolate import RegularGridInterpolator


def _array(name, values, *, lower=None, upper=None):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if lower is not None and np.any(values < lower):
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and np.any(values > upper):
        raise ValueError(f"{name} must be at most {upper}")
    return values


def _positive(name, values):
    values = _array(name, values)
    if np.any(values <= 0):
        raise ValueError(f"{name} must be positive")
    return values


def planck_um(wavelength_um, temperature_k):
    """Eq. (B.6), with the spectral axis last. Return W m^-2 sr^-1 um^-1."""
    wavelength_m = _positive("wavelength_um", wavelength_um) * 1e-6
    temperature_k = _positive("temperature_k", temperature_k)
    exponent = Planck * speed_of_light / (Boltzmann * temperature_k[..., None] * wavelength_m)
    with np.errstate(over="ignore"):
        return 2 * Planck * speed_of_light**2 * 1e-6 / (wavelength_m**5 * np.expm1(exponent))


@dataclass(frozen=True)
class SpectralGrid:
    """Common grid with separate response, integration limits and noise for WB and NB."""

    wavelength_um: np.ndarray
    response: np.ndarray
    bands_um: np.ndarray
    noise: np.ndarray

    def __post_init__(self):
        grid = _positive("wavelength_um", self.wavelength_um)
        response = _array("response", self.response, lower=0)
        bands = _positive("bands_um", self.bands_um)
        noise = _positive("noise", self.noise)
        if grid.ndim != 1 or grid.size < 3 or np.any(np.diff(grid) <= 0):
            raise ValueError("wavelength_um must be strictly increasing")
        if response.shape != (2, grid.size) or bands.shape != (2, 2) or noise.shape != (2,):
            raise ValueError("Expected response [2,L], bands_um [2,2] and noise [2] in WB/NB order")
        if not np.allclose(bands[0], (3.0, 5.0), rtol=0, atol=1e-12):
            raise ValueError("WB limits must be 3 and 5 micrometres")
        if not 3 <= bands[1, 0] < 3.9 < bands[1, 1] <= 5 or np.allclose(bands[0], bands[1]):
            raise ValueError("NB limits must enclose 3.9 micrometres in a separate band within WB")
        indices = np.searchsorted(grid, bands)
        if (indices >= grid.size).any() or not np.array_equal(grid[indices], bands):
            raise ValueError("Include each band endpoint on wavelength_um")
        maxima = response.max(axis=1)
        if np.all(maxima > 0) and np.allclose(response[0] / maxima[0], response[1] / maxima[1]):
            raise ValueError("WB and NB must be independently calibrated response curves")
        slices = tuple(slice(left, right + 1) for left, right in indices)
        weights = np.zeros_like(response)
        basis = np.zeros_like(response)
        for channel, span in enumerate(slices):
            segment = grid[span]
            signal = response[channel, span]
            if np.count_nonzero(signal) < 2:
                raise ValueError("Each channel needs a measured response on multiple in-band wavelengths")
            dx = np.diff(segment) / 2
            weights[channel, span] = signal * (np.r_[dx, 0] + np.r_[0, dx])
            basis[channel, span] = signal / weights[channel].sum()
        matrix = weights @ basis.T
        if np.linalg.cond(matrix) > 1e10:
            raise ValueError("WB and NB responses do not provide an independent background inversion")
        for field, value in (("wavelength_um", grid), ("response", response), ("bands_um", bands),
                             ("noise", noise), ("_weights", weights), ("_basis", basis), ("_matrix", matrix)):
            object.__setattr__(self, field, value)

    def integrate(self, radiance, channel):
        """Unnormalised response-weighted integration, Eq. (17)."""
        radiance = np.asarray(radiance, dtype=float)
        if radiance.shape[-1:] != self.wavelength_um.shape or channel not in (0, 1):
            raise ValueError("Spectral integration needs a final axis L and a WB/NB channel index")
        return radiance @ self._weights[channel]

    def background(self, sea_channel_means):
        """Eq. (B.7)-(B.8), retaining the signed two-basis fit."""
        means = _array("sea_channel_means", sea_channel_means, lower=0)
        if means.shape != (2,):
            raise ValueError("sea_channel_means must contain WB and NB means")
        return np.linalg.solve(self._matrix, means) @ self._basis

    def equivalent_solar(self, solar_irradiance):
        """Eq. (D.2). Input and output spectral density are W m^-2 um^-1."""
        irradiance = _array("solar_irradiance", solar_irradiance, lower=0)
        if irradiance.shape != self.wavelength_um.shape:
            raise ValueError("solar_irradiance must follow wavelength_um")
        return (self._weights @ irradiance) / self._weights.sum(axis=1)


def classify_los(surface_distance_m, plume_entry_m):
    """Eq. (B.5), assigning a finite tie to the opaque surface."""
    surface_distance_m = np.asarray(surface_distance_m)
    plume_entry_m = np.asarray(plume_entry_m)
    if surface_distance_m.shape != plume_entry_m.shape:
        raise ValueError("Surface and plume distance maps must have the same shape")
    for name, distance in (("surface_distance_m", surface_distance_m), ("plume_entry_m", plume_entry_m)):
        if np.any(np.isnan(distance) | (distance < 0)):
            raise ValueError(f"{name} must contain forward distances or +inf")
    surface = np.isfinite(surface_distance_m) & (surface_distance_m <= plume_entry_m)
    plume = np.isfinite(plume_entry_m) & (plume_entry_m < surface_distance_m)
    return surface, plume, ~(surface | plume)


def surface_radiance(wavelength_um, temperature_k, emissivity, background):
    """Eq. (5), using emissivity with an explicit last spectral axis."""
    emissivity = _array("surface_emissivity", emissivity, lower=0, upper=1)
    return emissivity * planck_um(wavelength_um, temperature_k) + (1 - emissivity) * background


def plume_radiance(wavelength_um, temperatures_k, absorption_m_inv, thickness_m):
    """Eq. (6). Layers on the penultimate axis are ordered far to near."""
    temperatures_k = _positive("plume temperatures_k", temperatures_k)
    absorption = _array("plume absorption_m_inv", absorption_m_inv, lower=0)
    thickness = _array("plume thickness_m", thickness_m, lower=0)
    if temperatures_k.ndim != 1 or not temperatures_k.size:
        raise ValueError("plume temperatures_k must be a nonempty layer vector")
    if absorption.shape != (temperatures_k.size, np.size(wavelength_um)) or thickness.shape[-1:] != temperatures_k.shape:
        raise ValueError("Plume absorption needs shape [J,L] and thickness a final layer axis J")
    optical_depth = thickness[..., None] * absorption
    foreground_depth = np.zeros_like(optical_depth)
    foreground_depth[..., :-1, :] = np.cumsum(optical_depth[..., :0:-1, :], axis=-2)[..., ::-1, :]
    emitted = planck_um(wavelength_um, temperatures_k) * -np.expm1(-optical_depth)
    return np.sum(emitted * np.exp(-foreground_depth), axis=-2)


@dataclass(frozen=True)
class Atmosphere:
    """Distinct atmospheric terms, either modelled or drawn from a named external boundary."""

    transmittance: np.ndarray
    path_radiance: np.ndarray
    source: str

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("atmosphere.source must identify Kruse or the external table")
        object.__setattr__(self, "transmittance", _array("atmosphere.transmittance", self.transmittance, lower=0, upper=1))
        object.__setattr__(self, "path_radiance", _array("atmosphere.path_radiance", self.path_radiance, lower=0))


def kruse_atmosphere(wavelength_um, visibility_km, path_m, air_temperature_k):
    """Eqs. (12)-(13), (D.1). Valid for 6–50 km meteorological visibility."""
    wavelength = _positive("wavelength_um", wavelength_um)
    visibility = _array("visibility_km", visibility_km, lower=6, upper=50)
    path_km = _array("path_m", path_m, lower=0) / 1000
    depth = (3.912 * path_km / visibility)[..., None] * (wavelength / 0.55) ** -1.3
    return Atmosphere(np.exp(-depth), -np.expm1(-depth) * planck_um(wavelength, air_temperature_k), "kruse:D.1")


def sea_radiance(wavelength_um, sea_temperature_k, sea_emissivity, reflectance, glint_weight, equivalent_solar):
    """Eq. (14). Directional reflectance is supplied by the sea BRDF calibration."""
    emissivity = _array("sea_emissivity", sea_emissivity, lower=0, upper=1)
    reflection = _array("sea_reflectance", reflectance, lower=0)
    glint = _array("sea_glint_weight", glint_weight, lower=0, upper=1)
    irradiance = _array("equivalent_solar", equivalent_solar, lower=0)
    return emissivity * planck_um(wavelength_um, sea_temperature_k) + (glint * reflection)[..., None] * irradiance / np.pi


def sea_directional_reflectance(rays, sun_direction, sea_normal, incident_zenith_deg,
                                emergent_zenith_deg, directional_factor):
    """Interpolate a measured directional BRDF factor for Eq. (14), without extrapolation."""
    rays = _array("sea rays", rays)
    sun = _array("sun_direction", sun_direction)
    normal = _array("sea_normal", sea_normal)
    if rays.shape[-1:] != (3,) or sun.shape != (3,) or normal.shape != (3,):
        raise ValueError("Sea rays, Sun direction and sea normal must have three vector components")
    for name, vector in (("sea rays", rays), ("Sun direction", sun), ("sea normal", normal)):
        if not np.allclose(np.linalg.norm(vector, axis=-1), 1, atol=1e-7):
            raise ValueError(f"{name} must be unit vectors")
    axes = tuple(_array(name, values, lower=0, upper=90) for name, values in
                 (("incident_zenith_deg", incident_zenith_deg), ("emergent_zenith_deg", emergent_zenith_deg)))
    if any(axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0) for axis in axes):
        raise ValueError("BRDF calibration needs strictly increasing incident and emergent angle axes")
    factors = _array("directional_factor", directional_factor, lower=0)
    if factors.shape != (axes[0].size, axes[1].size):
        raise ValueError("directional_factor must have shape [incident angle, emergent angle]")
    interpolation = RegularGridInterpolator(axes, factors, bounds_error=True)
    mu_sun = sun @ normal
    if mu_sun <= 0:
        return np.zeros(rays.shape[:-1])
    mu_sensor = -rays @ normal
    if np.any(mu_sensor <= 0):
        raise ValueError("Visible sea rays must point down towards the mean sea surface")
    incident = np.rad2deg(np.arccos(np.clip(mu_sun, 0, 1)))
    emergent = np.rad2deg(np.arccos(np.clip(mu_sensor, 0, 1)))
    query = np.stack([np.full_like(emergent, incident), emergent], axis=-1)
    return interpolation(query).reshape(emergent.shape)
