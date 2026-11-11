"""Pixel-domain visibility and the projected, nozzle-local VSO search."""

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Tuple

import numpy as np

from .nozzles import Action


J_LIM = 3.0
BETA_W = 0.10
BETA_U = 0.10
N_VSO = 3
ETA = 0.01
DELTA_FD = 0.05
EPS_U = 0.05


def visibility(radiance, masks, valid, noise, glint):
    """Eqs. (23)–(24), one score per spray ROI in a co-registered image.

    Channel order is 3–5 µm then 3.9 µm. Noise is expressed in the
    same band-integrated radiance units as the frames.
    """
    frames = np.asarray(radiance, dtype=float)
    if frames.ndim != 3 or frames.shape[0] != 2:
        raise ValueError("radiance must have shape [2, H, W]")
    good = np.asarray(valid)
    if good.dtype.kind != "b" or good.shape not in (frames.shape[1:], frames.shape):
        raise ValueError("valid must be Boolean with shape [H, W] or [2, H, W]")
    good = good & np.isfinite(frames)
    noise = np.asarray(noise, dtype=float)
    if noise.shape != (2,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("noise must give both positive NETD-equivalent radiance amplitudes")
    if not np.isfinite(glint) or not 0 <= glint <= 1:
        raise ValueError("glint must be finite and in [0,1]")

    def pixels(region):
        if region not in masks:
            raise ValueError(f"Missing {region} ROI")
        mask = np.asarray(masks[region])
        if mask.dtype.kind != "b" or mask.shape != frames.shape[1:]:
            raise ValueError(f"{region} ROI must be a Boolean image mask")
        return [frame[ok & mask] for frame, ok in zip(frames, good)]

    sea = pixels("Sea")
    if any(channel.size < 2 for channel in sea):
        raise ValueError("Visibility requires at least two valid Sea pixels per channel")
    sea_mean = np.array([channel.mean() for channel in sea])
    sea_std = np.array([channel.std(ddof=1) for channel in sea])
    denominator = np.maximum(sea_std, noise)
    scores = {}
    for region in masks:
        if region in ("Sea", "Global"):
            continue
        target = pixels(region)
        if any(channel.size == 0 for channel in target):
            raise ValueError(f"Visibility requires valid pixels in both channels of ROI {region}")
        target_mean = np.array([channel.mean() for channel in target])
        contrast = np.maximum(target_mean - sea_mean, 0) / denominator
        scores[region] = float((1 - glint) * contrast[0] + glint * contrast[1])
    return scores


def finite_difference(function, point, step=DELTA_FD):
    """Symmetric feasible probes with a one-sided stencil at either bound."""
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all() or np.any((point < 0) | (point > 1)):
        raise ValueError("Finite differences require a point in the normalised unit cube")
    if not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError("Finite difference step must be in (0, 1]")
    gradient = np.empty(3)
    for coordinate in range(3):
        left, right = point.copy(), point.copy()
        left[coordinate] = max(0.0, left[coordinate] - step)
        right[coordinate] = min(1.0, right[coordinate] + step)
        gradient[coordinate] = (function(right) - function(left)) / (
            right[coordinate] - left[coordinate]
        )
    return gradient


@dataclass(frozen=True)
class LocalCandidate:
    values: Tuple[float, float, float]
    normalised: Tuple[float, float, float]
    visibility: float
    cost: float


@dataclass(frozen=True)
class NozzleSearch:
    nozzle_id: str
    region: str
    iterates: Tuple[LocalCandidate, ...]
    probes: Tuple[LocalCandidate, ...]
    best: LocalCandidate
    selected: LocalCandidate
    reason: str

    @property
    def iterations(self):
        return len(self.iterates) - 1


@dataclass(frozen=True)
class Selection:
    raw: Optional[np.ndarray]
    projected: Action
    projected_scores: Dict[str, float]
    searches: Tuple[NozzleSearch, ...]
    final: Action
    final_scores: Dict[str, float]
    evaluations: int
    guard_ms: float
    projected_evaluation_ms: float
    vso_ms: float


def final_choice(initial, best):
    """Eq. (27), including the inclusive threshold and tolerance."""
    if best.visibility <= J_LIM:
        return best, "threshold_met"
    correction = np.asarray(best.normalised) - initial.normalised
    if np.linalg.norm(correction) <= EPS_U:
        return initial, "within_tolerance"
    return best, "best_correction"


def select_action(raw, registry, evaluator, *, enabled):
    """Eqs. (22)–(27), with each nozzle substituted into the projected set.

    The evaluator receives a complete Action and returns regional scores.
    Search probes contribute gradients. The argmin in Eq. (26) is over
    the projected initial candidate and the successive search iterates.
    """
    start = perf_counter()
    if raw is not None:
        raw = np.array(raw, dtype=float, copy=True)
        raw.setflags(write=False)
    projected = registry.project(raw, enabled)
    guard_ms = 1000 * (perf_counter() - start)
    final_values = projected.values.copy()
    evaluations = 0
    active_regions = {region for region, on in zip(registry.regions, projected.enabled) if on}

    def assess(action):
        nonlocal evaluations
        scores = dict(evaluator(action))
        evaluations += 1
        for region in active_regions:
            if region not in scores or not np.isfinite(scores[region]) or scores[region] < 0:
                raise ValueError(f"Candidate evaluation has no valid non-negative score for ROI {region}")
        return {
            region: float(scores[region]) for region in registry.region_ids
            if region in scores and np.isfinite(scores[region]) and scores[region] >= 0
        }

    start = perf_counter()
    projected_scores = assess(projected)
    projected_evaluation_ms = 1000 * (perf_counter() - start)
    start = perf_counter()
    searches = []
    for index, nozzle in enumerate(registry.nozzles):
        if not projected.enabled[index]:
            continue
        initial_visibility = projected_scores[nozzle.region]
        if initial_visibility <= J_LIM:
            continue
        low = registry.lower[index]
        high = registry.upper[index]
        span = high - low
        initial = np.clip((projected.values[index] - low) / span, 0, 1)
        reference_flow = nozzle.flow_l_min(nozzle.reference_pressure_mpa, 1.0)
        cache = {}
        probes = []

        def evaluate(normalised, known_visibility: Optional[float] = None):
            key = tuple(float(v) for v in normalised)
            if key in cache:
                return cache[key]
            values = np.clip(low + normalised * span, low, high)
            flow = nozzle.flow_l_min(values[0], values[1])
            if known_visibility is None:
                substituted = projected.values.copy()
                substituted[index] = values
                known_visibility = assess(Action(substituted, projected.enabled))[nozzle.region]
            deviation = normalised - initial
            cost = known_visibility + BETA_W * flow / reference_flow + BETA_U * (deviation @ deviation)
            candidate = LocalCandidate(tuple(float(v) for v in values), key, known_visibility, float(cost))
            cache[key] = candidate
            return candidate

        def probe(normalised):
            candidate = evaluate(normalised)
            probes.append(candidate)
            return candidate.cost

        current = initial
        best = initial_candidate = evaluate(initial, initial_visibility)
        iterates = [initial_candidate]
        for _ in range(N_VSO):
            gradient = finite_difference(probe, current)
            current = np.clip(current - ETA * gradient, 0, 1)
            candidate = evaluate(current)
            iterates.append(candidate)
            if candidate.cost < best.cost:
                best = candidate
            if best.visibility <= J_LIM:
                break
        selected, reason = final_choice(initial_candidate, best)
        if selected is not initial_candidate:
            final_values[index] = selected.values
        searches.append(
            NozzleSearch(nozzle.nozzle_id, nozzle.region, tuple(iterates), tuple(probes), best, selected, reason)
        )

    final = Action(final_values, projected.enabled)
    final_scores = projected_scores
    if not np.array_equal(final.values, projected.values):
        final_scores = assess(final)
    vso_ms = 1000 * (perf_counter() - start) if searches else 0.0
    start = perf_counter()
    registry.validate_action(final, feasible=True)
    guard_ms += 1000 * (perf_counter() - start)
    return Selection(
        raw, projected, projected_scores, tuple(searches), final, final_scores,
        evaluations, guard_ms, projected_evaluation_ms, vso_ms,
    )
