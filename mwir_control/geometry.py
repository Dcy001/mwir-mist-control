"""Camera, vessel and line-of-sight geometry, Appendix B.1–B.2."""

from dataclasses import dataclass

import numpy as np


def orthogonal(matrix, field, *, proper=False):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7):
        raise ValueError(f"{field} must be a 3×3 orthogonal matrix")
    if proper and not np.isclose(np.linalg.det(matrix), 1.0):
        raise ValueError(f"{field} must have determinant +1")
    return matrix


def vec3(value, field):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{field} must contain three finite components")
    return value


def heading_basis(heading_deg):
    """Map forward/starboard/up components to ENU at zero roll and pitch."""
    if not np.isfinite(heading_deg):
        raise ValueError("heading_deg must be finite")
    heading = np.deg2rad(heading_deg)
    return np.array([[np.sin(heading), np.cos(heading), 0],
                     [np.cos(heading), -np.sin(heading), 0],
                     [0, 0, 1]])


def observer_offset(forward, starboard, quadrant):
    """Unit horizontal observer offset using the actual positive starboard axis."""
    signs = {"lf": (1, -1), "rf": (1, 1), "lr": (-1, -1), "rr": (-1, 1)}
    if quadrant not in signs:
        raise ValueError(f"Unknown observer quadrant {quadrant!r}")
    axes = np.stack([vec3(forward, "forward"), vec3(starboard, "starboard")])
    if not np.allclose(axes @ axes.T, np.eye(2), atol=1e-7):
        raise ValueError("Forward and starboard must be orthonormal")
    return np.asarray(signs[quadrant]) @ axes / np.sqrt(2)


@dataclass(frozen=True)
class Camera:
    shape: tuple
    fx: float
    fy: float
    cx: float
    cy: float
    origin_m: np.ndarray
    camera_to_enu: np.ndarray

    def __post_init__(self):
        if len(self.shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in self.shape):
            raise ValueError("camera.shape must be (height, width) with positive integers")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("camera intrinsics must be finite and focal lengths positive")
        object.__setattr__(self, "origin_m", vec3(self.origin_m, "camera.origin_m"))
        object.__setattr__(self, "camera_to_enu", orthogonal(self.camera_to_enu, "camera.camera_to_enu", proper=True))

    def rays(self):
        """Unit rays from zero-based pixel centres, shape (height, width, 3)."""
        y, x = np.indices(self.shape)
        directions = np.stack([(x - self.cx) / self.fx, (y - self.cy) / self.fy, np.ones(self.shape)], axis=-1)
        directions = directions @ self.camera_to_enu.T
        return directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    def project(self, points_enu):
        """Project ENU points in front of the optical centre to (x, y) pixels."""
        points = np.asarray(points_enu, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("Projection requires finite ENU points (..., 3)")
        local = (points - self.origin_m) @ self.camera_to_enu
        if (local[..., 2] <= 0).any():
            raise ValueError("ROI geometry must be clipped to the front of the camera before projection")
        return local[..., :2] / local[..., 2, None] * [self.fx, self.fy] + [self.cx, self.cy]


def solar_unit_enu(elevation_deg, azimuth_deg):
    """Direction to the Sun, azimuth clockwise from true north."""
    if not np.isfinite([elevation_deg, azimuth_deg]).all() or abs(elevation_deg) > 90:
        raise ValueError("Solar elevation must be in [-90, 90] degrees and azimuth finite")
    elevation, azimuth = np.deg2rad([elevation_deg, azimuth_deg])
    return np.array([np.cos(elevation) * np.sin(azimuth),
                     np.cos(elevation) * np.cos(azimuth), np.sin(elevation)])


def _slab(origin, rays, lower, upper):
    parallel = np.abs(rays) < 1e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lower - origin) / rays
        t1 = (upper - origin) / rays
    outside = ((origin < lower) | (origin > upper)) & parallel
    entry = np.maximum(0, np.max(np.where(parallel, -np.inf, np.minimum(t0, t1)), axis=-1))
    exit = np.min(np.where(parallel, np.inf, np.maximum(t0, t1)), axis=-1)
    return entry, exit, outside.any(axis=-1)


def ray_box_interval(origin_m, rays, lower_m, upper_m):
    """Forward AABB segment, with (+inf, +inf) on miss or zero-length contact."""
    origin = vec3(origin_m, "origin_m")
    lower, upper = vec3(lower_m, "lower_m"), vec3(upper_m, "upper_m")
    directions = np.asarray(rays, dtype=float)
    if (lower >= upper).any():
        raise ValueError("The box must have positive extent along every axis")
    _validate_rays(directions)
    entry, exit, outside = _slab(origin, directions, lower, upper)
    valid = (exit > entry) & ~outside
    return np.where(valid, entry, np.inf), np.where(valid, exit, np.inf)


def _validate_rays(directions):
    if directions.ndim < 1 or directions.shape[-1] != 3 or not np.allclose(
        np.linalg.norm(directions, axis=-1), 1.0, atol=1e-7
    ):
        raise ValueError("Rays must be unit vectors with shape (..., 3)")


@dataclass(frozen=True)
class RayHits:
    distance_m: np.ndarray
    triangle_index: np.ndarray
    normal: np.ndarray


@dataclass
class _Node:
    lower: np.ndarray
    upper: np.ndarray
    indices: np.ndarray
    children: tuple


class TriangleScene:
    """Two-sided opaque mesh with a bounding volume hierarchy and batched rays."""

    def __init__(self, vertices_enu_m, triangle_indices):
        vertices = np.asarray(vertices_enu_m, dtype=float)
        triangles = np.asarray(triangle_indices)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("vertices_enu_m must be a finite (V, 3) array")
        if (triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu"
                or (triangles < 0).any() or (triangles >= len(vertices)).any()):
            raise ValueError("triangle_indices must be an in-range integer (F, 3) array")
        corners = vertices[triangles]
        self.a = corners[:, 0]
        self.edge1 = corners[:, 1] - self.a
        self.edge2 = corners[:, 2] - self.a
        normals = np.cross(self.edge1, self.edge2)
        length = np.linalg.norm(normals, axis=-1)
        self.normals = np.divide(normals, length[:, None], out=np.zeros_like(normals), where=length[:, None] > 0)
        lower, upper = corners.min(axis=1), corners.max(axis=1)
        centre = (lower + upper) / 2

        def build(indices):
            bounds = lower[indices].min(axis=0), upper[indices].max(axis=0)
            if len(indices) <= 12:
                return _Node(*bounds, indices, ())
            axis = int(np.argmax(np.ptp(centre[indices], axis=0)))
            order = indices[np.argsort(centre[indices, axis], kind="stable")]
            midpoint = len(order) // 2
            return _Node(*bounds, np.empty(0, dtype=int), (build(order[:midpoint]), build(order[midpoint:])))

        # Some CAD exports include zero-area facets. Keep their original indices.
        usable = np.flatnonzero(length > 0)
        self.root = build(usable) if usable.size else None

    def intersect(self, origin_m, rays):
        origin = vec3(origin_m, "origin_m")
        directions = np.asarray(rays, dtype=float)
        _validate_rays(directions)
        shape = directions.shape[:-1]
        flat = directions.reshape(-1, 3)
        distance = np.full(len(flat), np.inf)
        hit = np.full(len(flat), -1, dtype=int)
        pending = [(self.root, np.arange(len(flat)))] if self.root else []
        while pending:
            node, indices = pending.pop()
            entry, exit, outside = _slab(origin, flat[indices], node.lower - 1e-9, node.upper + 1e-9)
            indices = indices[(exit >= entry) & (entry <= distance[indices]) & ~outside]
            if not indices.size:
                continue
            if node.children:
                pending.extend((child, indices) for child in node.children)
                continue
            faces = node.indices
            e1, e2 = self.edge1[faces], self.edge2[faces]
            # Möller–Trumbore barycentric intersection inside each finite facet.
            p = np.cross(flat[indices, None, :], e2[None, :, :])
            det = np.einsum("nfi,fi->nf", p, e1)
            inverse = np.divide(1, det, out=np.zeros_like(det), where=np.abs(det) > 1e-13)
            tvec = origin - self.a[faces]
            u = np.einsum("fi,nfi->nf", tvec, p) * inverse
            q = np.cross(tvec, e1)
            v = (flat[indices] @ q.T) * inverse
            t = np.einsum("fi,fi->f", e2, q) * inverse
            valid = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > 1e-9)
            values = np.where(valid, t, np.inf)
            best = np.argmin(values, axis=1)
            nearest = values[np.arange(len(indices)), best]
            nearer = nearest < distance[indices]
            distance[indices[nearer]] = nearest[nearer]
            hit[indices[nearer]] = faces[best[nearer]]
        normal = np.zeros_like(flat)
        valid = hit >= 0
        normal[valid] = self.normals[hit[valid]]
        flip = np.einsum("ij,ij->i", flat, normal) > 0
        normal[flip] *= -1
        return RayHits(distance.reshape(shape), hit.reshape(shape), normal.reshape((*shape, 3)))
