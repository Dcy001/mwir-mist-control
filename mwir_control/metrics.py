"""Evaluation metrics from Appendix H, computed on declared matched support."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .control import visibility
from .nozzles import SC_REGION_IDS
from .roi import hotspot_statistics


FULL_LOAD_SECONDS = 24 * 60 * 60
LATENCY_STAGES = ("io", "phys", "prep", "lstm", "vso", "guard", "tx")
STUDY_SPRAY_ROI_IDS = ("SCC-F", "SCC-A", *SC_REGION_IDS)
STUDY_ROI_IDS = ("Sea", "Global", *STUDY_SPRAY_ROI_IDS)


def _clock(hour, minute):
    return 3600 * hour + 60 * minute


WINDOWS_S = {
    "SCC-F": (_clock(7, 14), _clock(21, 14)),
    "SCC-A": (_clock(7, 54), _clock(20, 34)),
    "SC": (_clock(6, 54), _clock(22, 14)),
}


def _finite(value, name):
    result = np.asarray(value, dtype=float)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def prediction_metrics(predicted, labels, lower, upper, *, target_enabled=None):
    """Eqs. (H.1)–(H.2), over enabled targets for each nozzle and component.

    If target_enabled is omitted, every row must already be an enabled pair.
    A nozzle with no enabled pairs has undefined metrics.
    """
    predicted = np.asarray(predicted, dtype=float)
    labels = np.asarray(labels, dtype=float)
    lower = _finite(lower, "lower bounds")
    upper = _finite(upper, "upper bounds")
    if (
        not labels.size or predicted.shape != labels.shape or labels.ndim != 3 or labels.shape[2] != 3
        or lower.shape != labels.shape[1:] or upper.shape != lower.shape or np.any(lower >= upper)
    ):
        raise ValueError("Prediction/label arrays must share [T,N,3] and bounds [N,3]")
    enabled = np.ones(labels.shape[:2], dtype=bool) if target_enabled is None else np.asarray(target_enabled)
    if enabled.shape != labels.shape[:2] or enabled.dtype.kind != "b":
        raise ValueError("Prediction target_enabled must be Boolean [T,N]")
    if not np.isfinite(predicted[enabled]).all() or not np.isfinite(labels[enabled]).all():
        raise ValueError("Prediction and label values must be finite for every enabled target")
    active = enabled[..., None]
    count = enabled.sum(axis=0)
    denominator = np.where(count[:, None] > 0, count[:, None], np.nan)
    predicted = np.where(active, predicted, 0)
    labels = np.where(active, labels, 0)
    error = predicted - labels
    squared = error ** 2
    mean = labels.sum(axis=0) / np.maximum(count[:, None], 1)
    variance_sum = np.sum(np.where(active, (labels - mean) ** 2, 0), axis=0)
    r2 = np.full(variance_sum.shape, np.nan)
    np.divide(squared.sum(axis=0), variance_sum, out=r2, where=variance_sum > 0)
    return {
        "evaluated_count": count,
        "mae": np.abs(error).sum(axis=0) / denominator,
        "rmse": np.sqrt(squared.sum(axis=0) / denominator),
        "r2": 1 - r2,
        "cvr_pct": 100 * (active & ((predicted < lower) | (predicted > upper))).sum(axis=0) / denominator,
    }


@dataclass(frozen=True)
class Trial:
    """Aligned record with a stable camera-grid identity.

    Arrays: timestamps_ns [T], radiance [T,2,H,W], valid [T,H,W]
    or [T,2,H,W], and optional temperature [T,H,W] (°C or K).
    """

    timestamps_ns: np.ndarray
    grid_id: str
    radiance: np.ndarray
    valid: np.ndarray
    temperature: np.ndarray = None


def _normalise_trial(trial):
    t = np.asarray(trial.timestamps_ns)
    frames = np.asarray(trial.radiance)
    good = np.asarray(trial.valid)
    if not trial.grid_id:
        raise ValueError("Trial grid_id is required")
    if t.ndim != 1 or not t.size or t.dtype.kind != "i" or np.any(t[1:] <= t[:-1]):
        raise ValueError("Trial timestamps_ns must be unique increasing signed integers")
    if frames.ndim != 4 or frames.shape[:2] != (len(t), 2):
        raise ValueError("Trial radiance must be [T,2,H,W]")
    if good.dtype.kind != "b":
        raise ValueError("Trial validity must be Boolean")
    temperature = None if trial.temperature is None else np.asarray(trial.temperature)
    if temperature is not None and temperature.shape != (len(t),) + frames.shape[2:]:
        raise ValueError("Trial temperature must be [T,H,W]")
    if good.shape == (len(t),) + frames.shape[2:]:
        good = good[:, None, :, :]
    if good.shape not in (frames.shape, (len(t), 1) + frames.shape[2:]):
        raise ValueError("Trial validity must be [T,H,W] or [T,2,H,W]")
    return t, frames, good, temperature


def _percentage_drop(baseline, strategy, name):
    if baseline <= 0:
        raise ValueError(f"{name}: baseline must be positive")
    return float(100 * (baseline - strategy) / baseline)


def _highlight(frame, mask, sea, valid, area):
    pixels = frame[mask & valid]
    if pixels.size == 0:
        raise ValueError("3.9 µm highlight ROI contains no valid pixel")
    top_n = int(np.ceil(0.01 * pixels.size))
    hotspot = hotspot_statistics(frame, mask, sea, valid, area)
    support_area = area * np.count_nonzero(mask)
    if support_area <= 0 or not np.isfinite(hotspot.area_m2):
        raise ValueError("3.9 µm hotspot needs positive ROI area and a valid Sea threshold")
    return np.array([
        np.percentile(pixels, 95), pixels.max(),
        np.partition(pixels, pixels.size - top_n)[-top_n:].mean(),
        100 * hotspot.area_m2 / support_area,
    ])


def _validate_roi_scope(masks, scope):
    scope = {"mode": "full"} if scope is None else scope
    if not isinstance(scope, Mapping) or scope.get("mode") not in ("full", "subset"):
        raise ValueError("comparison.roi_scope.mode must be 'full' or 'subset'")
    mode = scope["mode"]
    canonical = set(STUDY_ROI_IDS)
    if mode == "full":
        if set(scope) != {"mode"}:
            raise ValueError("Full comparison roi_scope takes only mode='full'")
        expected = canonical
        subset_id = None
    else:
        if set(scope) != {"mode", "subset_id", "roi_ids"}:
            raise ValueError("Subset comparison roi_scope requires mode, subset_id and roi_ids")
        subset_id, roi_ids = scope["subset_id"], scope["roi_ids"]
        if not isinstance(subset_id, str) or not subset_id.strip():
            raise ValueError("Subset comparison requires a nonempty subset_id")
        if not isinstance(roi_ids, (list, tuple)) or not all(isinstance(name, str) for name in roi_ids):
            raise ValueError("Subset comparison roi_ids must be a list of ROI names")
        expected = set(roi_ids)
        if len(expected) != len(roi_ids):
            raise ValueError("Subset comparison roi_ids contain duplicate names")
        unknown = expected - canonical
        if unknown:
            raise ValueError(f"Unknown subset ROI IDs: {sorted(unknown)}")
        if "Sea" not in expected or not expected.intersection(STUDY_SPRAY_ROI_IDS):
            raise ValueError("Subset comparison requires Sea and at least one canonical spray ROI")
        if expected == canonical:
            raise ValueError("Use full comparison mode for the complete study ROI list")
    actual = set(masks)
    missing = [name for name in STUDY_ROI_IDS if name in expected - actual]
    unknown = sorted(actual - canonical, key=str)
    undeclared = [name for name in STUDY_ROI_IDS if name in actual - expected]
    problems = []
    if missing:
        problems.append(f"missing ROI masks: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown ROI names: {', '.join(map(str, unknown))}")
    if undeclared:
        problems.append(f"undeclared ROI masks: {', '.join(undeclared)}")
    if problems:
        raise ValueError(f"{mode.capitalize()} comparison ROI scope: " + ". ".join(problems))
    roi_ids = [name for name in STUDY_ROI_IDS if name in expected]
    scope_info = {
        "mode": mode,
        "roi_ids": roi_ids,
        "coverage": {
            "evaluated_spray_rois": len(expected.intersection(STUDY_SPRAY_ROI_IDS)),
            "study_spray_rois": len(STUDY_SPRAY_ROI_IDS),
            "evaluated_sc_rois": len(expected.intersection(SC_REGION_IDS)),
            "study_sc_rois": len(SC_REGION_IDS),
        },
    }
    if subset_id is not None:
        scope_info["subset_id"] = subset_id
    return scope_info


def compare_trials(baseline, strategies, masks, pixel_area_m2, noise, glint, day_start_ns, epsilon,
                   *, roi_scope=None):
    """Eqs. (H.3)–(H.13), using the common support of all supplied strategies.

    Full evaluation requires all study ROIs. An explicit subset scope names
    the participating ROIs. Mask values are Boolean images [H,W] or a
    common mask sequence [T,H,W]. day_start_ns is local midnight on the
    trial's nanosecond time base.
    Temperature inputs share a single unit across the comparison.
    """
    if not strategies or "No-mist" in strategies:
        raise ValueError("Provide named treatment strategies separately from the No-mist baseline")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("H.7 epsilon must be an explicit positive value")
    if not isinstance(day_start_ns, (int, np.integer)) or isinstance(day_start_ns, bool):
        raise ValueError("day_start_ns must identify midnight in integer nanoseconds")
    trials = {"No-mist": baseline, **strategies}
    records = {name: _normalise_trial(trial) for name, trial in trials.items()}
    t, base_frames, _, _ = records["No-mist"]
    for name, trial in trials.items():
        timestamps, frames, _, _ = records[name]
        if trial.grid_id != baseline.grid_id or not np.array_equal(timestamps, t) or frames.shape != base_frames.shape:
            raise ValueError(f"{name} and No-mist must share timestamps and the same co-registered grid")
    scope = _validate_roi_scope(masks, roi_scope)
    sc_regions = [name for name in SC_REGION_IDS if name in masks]
    shape = (t.size,) + base_frames.shape[2:]
    masks = {name: np.asarray(mask) for name, mask in masks.items()}
    for name, mask in masks.items():
        if mask.dtype.kind != "b" or mask.shape not in (shape, shape[1:]):
            raise ValueError(f"{name} mask must be Boolean and match the common grid")
    area = _finite(pixel_area_m2, "pixel_area_m2")
    if area.shape not in ((), t.shape) or np.any(area <= 0):
        raise ValueError("Projected pixel area must be a positive scalar or a [T] array")
    area = np.broadcast_to(area, t.shape)
    glint = _finite(glint, "glint")
    if glint.shape != t.shape or np.any((glint < 0) | (glint > 1)):
        raise ValueError("glint must be in [0,1] for every matched timestamp")
    secs = (t - day_start_ns) / 1e9
    if np.any((secs < 0) | (secs >= FULL_LOAD_SECONDS)):
        raise ValueError("Trial timestamps must be in the declared 24 h experimental day")
    time_windows = {
        region: (secs >= start) & (secs <= end)
        for region, (start, end) in WINDOWS_S.items()
    }
    visibility_window = time_windows["SCC-F"] | time_windows["SCC-A"] | time_windows["SC"]
    time_windows["Global"] = visibility_window
    time_windows.update({name: time_windows["SC"] for name in sc_regions})
    active = {name: time_windows[name] for name in scope["roi_ids"] if name != "Sea"}
    missing_times = [region for region, window in active.items() if not window.any()]
    if missing_times:
        raise ValueError(f"No aligned timestamp in Appendix H spray-on windows for: {', '.join(missing_times)}")

    has_temperature = [data[3] is not None for data in records.values()]
    if any(has_temperature) and not all(has_temperature):
        raise ValueError("Temperature comparison requires all treatment and No-mist fields")
    means = {name: {region: [] for region in active} for name in trials}
    thermal_delta = {name: {region: [] for region in active} for name in strategies}
    maximum_visibility = {name: [] for name in trials}
    highlights = {name: {region: [] for region in sc_regions} for name in trials}

    for i in np.flatnonzero(visibility_window):
        masks_i = {name: mask[i] if mask.ndim == 3 else mask for name, mask in masks.items()}
        common = np.ones(base_frames.shape[1:], dtype=bool)
        for _, frames, good, _ in records.values():
            common &= good[i] & np.isfinite(frames[i])
        if all(has_temperature):
            temps = {name: data[3][i] for name, data in records.items()}
            common_temp = np.logical_and.reduce([np.isfinite(frame) for frame in temps.values()])
        for name, (_, frames, _, _) in records.items():
            try:
                scores = visibility(frames[i], masks_i, common, noise, glint[i])
            except ValueError as exc:
                raise ValueError(f"Comparison visibility at timestamp_ns={t[i]}: {exc}") from exc
            maximum_visibility[name].append(max(scores.values()))
            for region, window in active.items():
                if not window[i]:
                    continue
                support = masks_i[region] & common[0]
                if not support.any():
                    raise ValueError(f"ROI {region} has no shared main-channel support at {t[i]}")
                means[name][region].append(float(frames[i, 0][support].mean()))
                if region in sc_regions:
                    try:
                        highlight = _highlight(frames[i, 1], masks_i[region], masks_i["Sea"], common[1], area[i])
                    except ValueError as exc:
                        raise ValueError(f"ROI {region} highlight for {name} at timestamp_ns={t[i]}: {exc}") from exc
                    highlights[name][region].append(highlight)
                if name != "No-mist" and all(has_temperature):
                    good_temp = masks_i[region] & common_temp
                    if not good_temp.any():
                        raise ValueError(f"ROI {region} has no shared temperature support at {t[i]}")
                    thermal_delta[name][region].append(float(
                        np.mean(temps["No-mist"][good_temp] - temps[name][good_temp])
                    ))

    result = {}
    base_max = np.asarray(maximum_visibility["No-mist"])
    for name in strategies:
        residual = np.asarray(maximum_visibility[name]) / (base_max + epsilon)
        q95 = float(np.percentile(residual, 95))
        main = {}
        for region in active:
            base_mean = float(np.mean(means["No-mist"][region]))
            treatment_mean = float(np.mean(means[name][region]))
            main[region] = {
                "baseline_radiance": base_mean,
                "strategy_radiance": treatment_mean,
                "attenuation_pct": _percentage_drop(base_mean, treatment_mean, region),
                "timestamps": len(means[name][region]),
            }
            if all(has_temperature):
                main[region]["temperature_reduction_c"] = float(np.mean(thermal_delta[name][region]))
        aux = {}
        for region in sc_regions:
            values = np.asarray(highlights[name][region])
            average = values.mean(axis=0)
            base = np.mean(highlights["No-mist"][region], axis=0)
            aux[region] = {
                "q95_radiance": float(average[0]),
                "max_radiance_each_frame": values[:, 1],
                "timestamps_ns": t[active[region]],
                "top_1pct_radiance": float(average[2]),
                "highlight_area_pct": float(average[3]),
                "baseline_q95_radiance": float(base[0]),
                "baseline_top_1pct_radiance": float(base[2]),
                "baseline_highlight_area_pct": float(base[3]),
                "top_1pct_reduction_pct": _percentage_drop(base[2], average[2], region),
            }
        max_name = "subset_max" if scope["mode"] == "subset" else "max"
        result[name] = {
            "roi_scope": scope,
            "main_channel": main,
            "auxiliary_channel": aux,
            "visibility": {
                "timestamps_ns": t[visibility_window],
                f"baseline_{max_name}": base_max,
                f"strategy_{max_name}": np.asarray(maximum_visibility[name]),
                "residual": residual,
                "residual_q95": q95,
                "compression_pct": 100 * (1 - q95),
            },
        }
    return result


def water_metrics(flow_l_min, full_load_l_min, sample_seconds=1):
    """Eqs. (H.14)–(H.15), with the 24 h reference of §3.4."""
    flow = _finite(flow_l_min, "nozzle flow")
    capacity = _finite(full_load_l_min, "24 h full-load nozzle flow")
    if flow.ndim != 2 or flow.shape[1] != 286 or np.any(flow < 0):
        raise ValueError("Measured duty-averaged nozzle flow must be non-negative [T,286] in L/min")
    if capacity.shape not in ((286,), (FULL_LOAD_SECONDS, 286)) or np.any(capacity < 0):
        raise ValueError("Full-load flow must be non-negative [286] or a 24 h 1 Hz schedule [86400,286]")
    if not np.isfinite(sample_seconds) or sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if len(flow) * sample_seconds > FULL_LOAD_SECONDS:
        raise ValueError("Water consumption trial exceeds the 24 h reference period")
    volume = flow.sum() * sample_seconds / 60_000
    full_load = capacity.sum() / 60_000
    if capacity.ndim == 1:
        full_load *= FULL_LOAD_SECONDS
    if full_load <= 0:
        raise ValueError("The 24 h full-load reference must be positive")
    return {
        "water_m3": float(volume),
        "full_load_24h_m3": float(full_load),
        "saving_pct": float(100 * (1 - volume / full_load)),
    }


def latency_metrics(stage_ms):
    """Eqs. (H.16)–(H.19), latency in ms and a strict 1000 ms deadline."""
    if set(stage_ms) != set(LATENCY_STAGES):
        raise ValueError(f"Latency requires all seven aligned stages: {LATENCY_STAGES}")
    stages = [_finite(stage_ms[name], name) for name in LATENCY_STAGES]
    if any(x.shape != stages[0].shape or x.ndim != 1 or np.any(x < 0) for x in stages):
        raise ValueError("Latency stages must share a time axis and be non-negative")
    elapsed = np.sum(stages, axis=0)
    p50, p95, p99 = np.percentile(elapsed, [50, 95, 99])
    return {
        "per_cycle_ms": elapsed,
        "mean_ms": float(elapsed.mean()),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(elapsed.max()),
        "jitter_ms": float(elapsed.std(ddof=0)),
        "deadline_miss_pct": float(100 * (elapsed > 1000).mean()),
    }
