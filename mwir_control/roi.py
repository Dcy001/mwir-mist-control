"""Camera pixel regions, field statistics and glint geometry (Appendix A)."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage

from .geometry import solar_unit_enu as sun_direction_enu

CHANNELS = ("3-5", "3.9")
SEA = "Sea"
STACKS = ("SCC-F", "SCC-A")
_CROSS = ndimage.generate_binary_structure(2, 1)
_EIGHT = ndimage.generate_binary_structure(2, 2)


def _points(value, count, name):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < count:
        raise ValueError(f"{name} needs at least {count} projected (x, y) points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite projected coordinates")
    return points


def _rasterise(shape, points, radius, polygon):
    """Select pixel centres in the polygon or within radius of its edges."""
    height, width = shape
    left, top = np.maximum(0, np.floor(points.min(axis=0) - radius).astype(int))
    right, bottom = np.minimum(
        (width, height), np.ceil(points.max(axis=0) + radius).astype(int) + 1
    )
    mask = np.zeros(shape, dtype=bool)
    if left >= right or top >= bottom:
        return mask
    y, x = np.ogrid[top:bottom, left:right]
    inside = np.zeros((bottom - top, right - left), dtype=bool)
    near = inside.copy()
    ends = np.vstack((points[1:], points[:1])) if polygon else points[1:]
    for (ax, ay), (bx, by) in zip(points, ends):
        dx, dy = bx - ax, by - ay
        if polygon and dy != 0:
            crosses = (ay > y) != (by > y)
            inside ^= crosses & (x < ax + (y - ay) * dx / dy)
        length_sq = dx * dx + dy * dy
        fraction = 0.0
        if length_sq > 0:
            fraction = np.clip(((x - ax) * dx + (y - ay) * dy) / length_sq, 0, 1)
        dist_sq = (x - ax - fraction * dx) ** 2 + (y - ay - fraction * dy) ** 2
        near |= dist_sq <= radius * radius
    if not polygon:
        # The end cuts keep the band within its longitudinal segment.
        segments = np.diff(points, axis=0)
        moving = segments[np.any(segments != 0, axis=1)]
        if not len(moving):
            raise ValueError("a side centreline must have distinct projected points")
        ax, ay = points[0]
        bx, by = points[-1]
        near &= (x - ax) * moving[0, 0] + (y - ay) * moving[0, 1] >= 0
        near &= (x - bx) * moving[-1, 0] + (y - by) * moving[-1, 1] <= 0
    mask[top:bottom, left:right] = inside | near
    return mask


def build_masks(
    shape,
    stack_contours_xy: Mapping,
    side_centrelines_xy: Mapping,
    waterline_xy,
):
    """Rasterise Appendix A.2 from projected image geometry.

    Coordinates are (column, row) at integer pixel centres. Side polylines
    are already divided at the vessel's longitudinal region boundaries.
    The visible waterline is supplied as a single-valued polyline in x.
    """
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("image shape must be (height, width) with positive dimensions")
    if set(stack_contours_xy) != set(STACKS):
        raise ValueError("stack_contours_xy must contain SCC-F and SCC-A")
    if not side_centrelines_xy or any(
        not isinstance(name, str) or not name.startswith("SC_")
        or not name[3:].isdigit() or int(name[3:]) < 1
        for name in side_centrelines_xy
    ):
        raise ValueError("side_centrelines_xy must use SC_i region identities")
    masks = {
        name: _rasterise(shape, _points(contour, 3, name), 6, polygon=True)
        for name, contour in stack_contours_xy.items()
    }
    masks.update({
        name: _rasterise(shape, _points(line, 2, name), 8, polygon=False)
        for name, line in side_centrelines_xy.items()
    })
    waterline = _points(waterline_xy, 2, "waterline_xy")
    if waterline[0, 0] > waterline[-1, 0]:
        waterline = waterline[::-1]
    if np.any(np.diff(waterline[:, 0]) <= 0):
        raise ValueError("waterline_xy must be single-valued with monotone x")
    y, x = np.ogrid[:shape[0], :shape[1]]
    edge = np.interp(x, waterline[:, 0], waterline[:, 1], left=np.nan, right=np.nan)
    masks[SEA] = (y >= edge) & (y < edge + 20)
    return masks


@dataclass(frozen=True)
class Statistics:
    mean: float
    std: float
    maximum: float
    count: int


@dataclass(frozen=True)
class Hotspots:
    count: float
    area_m2: float
    mask: np.ndarray


def _support(field, mask, valid):
    field = np.asarray(field, dtype=float)
    mask = np.asarray(mask)
    valid = np.asarray(valid)
    if field.ndim != 2 or mask.shape != field.shape or valid.shape != field.shape:
        raise ValueError("field, ROI mask and validity must have the same [H,W] shape")
    if mask.dtype != np.bool_ or valid.dtype != np.bool_:
        raise ValueError("ROI mask and validity must be Boolean")
    return field, mask & valid & np.isfinite(field)


def roi_statistics(field, mask, valid):
    """Eq. (A.4) on valid, finite pixels with sample standard deviation."""
    field, support = _support(field, mask, valid)
    values = field[support]
    if not len(values):
        return Statistics(np.nan, np.nan, np.nan, 0)
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return Statistics(float(values.mean()), std, float(values.max()), len(values))


def hotspot_statistics(field, source_mask, sea_mask, valid, pixel_area_m2):
    """Eqs. (A.1)–(A.3), with a five-pixel projected minimum area."""
    area = float(pixel_area_m2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("pixel_area_m2 must be a positive finite projected pixel area")
    field, support = _support(field, source_mask, valid)
    background = roi_statistics(field, sea_mask, valid)
    if background.count < 2 or not support.any():
        return Hotspots(np.nan, np.nan, np.zeros_like(support))
    rows = np.flatnonzero(support.any(axis=1))
    columns = np.flatnonzero(support.any(axis=0))
    bounds = np.s_[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    region_support = support[bounds]
    seed = region_support & (field[bounds] >= background.mean + 3 * background.std)
    # The zero padding represents the exterior of the image during closing.
    closed = ndimage.binary_closing(np.pad(seed, 1), structure=_CROSS)[1:-1, 1:-1]
    labels, count = ndimage.label(closed & region_support, structure=_EIGHT)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    accepted = sizes >= 5
    accepted[0] = False
    mask = np.zeros_like(support)
    mask[bounds] = accepted[labels]
    return Hotspots(int(accepted.sum()), float(sizes[accepted].sum() * area), mask)


def summarise_rois(frame, masks, valid, pixel_area_m2):
    """Eq. (1) as [channel, ROI, statistic], with ROI order from masks."""
    frames, validity = np.asarray(frame), np.asarray(valid)
    if frames.ndim != 3 or frames.shape[0] != len(CHANNELS) or validity.shape != frames.shape:
        raise ValueError("ROI summaries require co-registered radiance and validity [2,H,W]")
    results = np.full((len(CHANNELS), len(masks), 5), np.nan)
    sea_mask = masks[SEA]
    for channel, (values, good) in enumerate(zip(frames, validity)):
        for index, (name, mask) in enumerate(masks.items()):
            stats = roi_statistics(values, mask, good)
            results[channel, index, :3] = stats.mean, stats.std, stats.maximum
            if name != SEA:
                hot = hotspot_statistics(values, mask, sea_mask, good, pixel_area_m2)
                results[channel, index, 3:] = hot.count, hot.area_m2
    return results


def glint_weights(rays_enu, solar_elevation_deg, solar_azimuth_deg, wind_speed_m_s):
    """Eqs. (2), (A.5) and (A.6) with rays pointing away from the camera."""
    rays = np.asarray(rays_enu, dtype=float)
    if rays.ndim < 2 or rays.shape[-1] != 3:
        raise ValueError("rays_enu must have shape [...,3]")
    if not np.allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-7):
        raise ValueError("rays_enu must be finite unit vectors")
    wind = float(wind_speed_m_s)
    if not np.isfinite(wind) or wind < 0:
        raise ValueError("wind_speed_m_s must be finite and non-negative")
    sun = sun_direction_enu(solar_elevation_deg, solar_azimuth_deg)
    if solar_elevation_deg <= 0:
        return np.zeros(rays.shape[:-1])
    bisector = sun - rays
    norm = np.linalg.norm(bisector, axis=-1)
    if np.any(norm < 1e-14):
        raise ValueError("glint half-vector undefined for antiparallel Sun and view directions")
    cosine = np.clip(bisector[..., 2] / norm, -1, 1)
    # tan²(arccos(cosine)), including a vanishing weight at grazing tilt.
    slope_sq = np.full(cosine.shape, np.inf)
    np.divide(1 - cosine ** 2, cosine ** 2, out=slope_sq, where=cosine != 0)
    return np.exp(-slope_sq / (2 * (0.003 + 0.00512 * wind)))


def glint_risk(rays_enu, sea_mask, solar_elevation_deg, solar_azimuth_deg, wind_speed_m_s):
    """Eq. (3) over the Sea ROI."""
    rays = np.asarray(rays_enu, dtype=float)
    sea = np.asarray(sea_mask)
    if rays.ndim != 3 or rays.shape[-1] != 3 or sea.shape != rays.shape[:2]:
        raise ValueError("rays_enu [H,W,3] and Sea mask [H,W] must agree")
    if sea.dtype != np.bool_ or not sea.any():
        raise ValueError("Sea mask must be Boolean and contain at least one pixel")
    return float(glint_weights(
        rays[sea], solar_elevation_deg, solar_azimuth_deg, wind_speed_m_s
    ).mean())
