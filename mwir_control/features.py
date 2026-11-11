"""Per-nozzle features from the paired observation and actual-state prediction."""
from typing import Mapping

import numpy as np

from .roi import CHANNELS, SEA, hotspot_statistics, roi_statistics


ENVIRONMENT_NAMES = ("air_temperature", "visibility", "wind_speed_10m", "sea_temperature")
VESSEL_NAMES = ("vessel_speed", "sin_heading", "cos_heading", "roll", "pitch")
ACTION_NAMES = ("pressure", "duty", "deflection")
CHANNEL_FEATURE_NAMES = (
    "observed_roi_mean", "residual_roi_mean", "residual_roi_std",
    "residual_hotspot_count", "residual_hotspot_area", "observed_sea_mean",
)
FEATURE_NAMES = (
    ENVIRONMENT_NAMES + VESSEL_NAMES + ACTION_NAMES + ("glint_risk",)
    + tuple(f"{band}_{name}" for band in CHANNELS for name in CHANNEL_FEATURE_NAMES)
)
FEATURE_UNITS = (
    "degC", "m", "m/s", "degC", "m/s", "1", "1", "deg", "deg",
    "MPa", "1", "deg", "1",
) + ("W/(m² sr)", "W/(m² sr)", "W/(m² sr)", "1", "m²", "W/(m² sr)") * 2
ACTION_UNITS = ("MPa", "1", "deg")
WINDOW_STEPS = 8


def residual_fields(observed, modelled):
    """Eq. (18), on band-integrated radiances of equal shape [2,H,W]."""
    observed = np.asarray(observed, dtype=float)
    modelled = np.asarray(modelled, dtype=float)
    if observed.ndim != 3 or observed.shape[0] != 2 or observed.shape != modelled.shape:
        raise ValueError("observed and modelled radiance must have shape [2,H,W]")
    residual = np.full(observed.shape, np.nan)
    valid = np.isfinite(observed) & np.isfinite(modelled)
    np.subtract(observed, modelled, out=residual, where=valid)
    return residual


def _block(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or np.isinf(value).any():
        raise ValueError(f"{name} must have shape {shape} with missing data marked NaN")
    return value


def build_features(
    observed, modelled, valid, masks: Mapping, pixel_area_m2, nozzle_regions,
    actual_actions, environment, vessel, glint,
):
    """Return Eq. (19) features in fixed registry order, shape [N,25].

    Supply environment [Ta_C, visibility_m, U10_m_s, Tsea_C], vessel
    [speed_m_s, heading_deg, roll_deg, pitch_deg] and executed actions [N,3].
    Channel order is 3–5 µm then 3.9 µm. Validity can be shared or per channel.
    """
    observed = np.asarray(observed, dtype=float)
    residual = residual_fields(observed, modelled)
    valid = np.asarray(valid)
    if valid.shape == observed.shape[1:]:
        valid = np.broadcast_to(valid, observed.shape)
    if valid.shape != observed.shape or valid.dtype != np.bool_:
        raise ValueError("valid must be Boolean with shape [H,W] or [2,H,W]")
    nozzle_regions = tuple(nozzle_regions)
    if not nozzle_regions or SEA in nozzle_regions:
        raise ValueError("nozzle_regions must contain only spray region identities")
    missing = (set(nozzle_regions) | {SEA}) - masks.keys()
    if missing:
        raise ValueError(f"missing ROI masks: {sorted(missing)}")
    environment = _block(environment, (4,), "environment")
    vessel = _block(vessel, (4,), "vessel")
    actual = _block(actual_actions, (len(nozzle_regions), 3), "actual_actions")
    glint = float(glint)
    if not (np.isnan(glint) or 0 <= glint <= 1):
        raise ValueError("glint must be in [0,1] or NaN")
    heading_rad = np.deg2rad(vessel[1])
    state = np.array([vessel[0], np.sin(heading_rad), np.cos(heading_rad), *vessel[2:]])

    features = np.empty((len(nozzle_regions), len(FEATURE_NAMES)))
    features[:, :4] = environment
    features[:, 4:9] = state
    features[:, 9:12] = actual
    features[:, 12] = glint
    regions = tuple(dict.fromkeys(nozzle_regions))
    region_position = {region: position for position, region in enumerate(regions)}
    index = np.array([region_position[region] for region in nozzle_regions])
    for channel, (obs, res, good) in enumerate(zip(observed, residual, valid)):
        sea_mean = roi_statistics(obs, masks[SEA], good).mean
        summaries = []
        for region in regions:
            mask = masks[region]
            observed_stats = roi_statistics(obs, mask, good)
            residual_stats = roi_statistics(res, mask, good)
            hot = hotspot_statistics(res, mask, masks[SEA], good, pixel_area_m2)
            summaries.append((
                observed_stats.mean, residual_stats.mean, residual_stats.std,
                hot.count, hot.area_m2, sea_mean,
            ))
        start = 13 + len(CHANNEL_FEATURE_NAMES) * channel
        features[:, start:start + len(CHANNEL_FEATURE_NAMES)] = np.asarray(summaries)[index]
    return features
