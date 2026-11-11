"""UTC millisecond pairing, drift correction, quality checks and repair."""
from dataclasses import dataclass

import numpy as np


STEP_MS = 1000
CHANNEL_JITTER_LIMIT_MS = 20


def utc_milliseconds(values, name="time_ms"):
    """Require explicit integer Unix milliseconds (UTC)."""
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.signedinteger):
        raise ValueError(f"{name} must be a 1-D array of signed UTC milliseconds")
    return array.astype(np.int64, copy=False)


def _strict_times(values, name):
    times = utc_milliseconds(values, name)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} must be strictly increasing within a record")
    return times


def _record_ids(values, size, name):
    records = np.asarray(values)
    if records.shape != (size,) or records.dtype.kind != "U" or np.any(records == ""):
        raise ValueError(f"{name} must contain one non-empty Unicode ID per timestamp")
    return records


def keyed_indices(reference_time, reference_id, source_time, source_id, require_all=True):
    """Find source rows by exact (record ID, timestamp) rather than row order."""
    source_time = utc_milliseconds(source_time, "source_time_ms")
    source_id = _record_ids(source_id, len(source_time), "source_record_id")
    reference_time = utc_milliseconds(reference_time, "reference_time_ms")
    reference_id = _record_ids(reference_id, len(reference_time), "reference_record_id")
    source_keys = list(zip(source_id, source_time))
    source_rows = {key: row for row, key in enumerate(source_keys)}
    if len(source_keys) != len(source_rows):
        raise ValueError("duplicate (source_record_id, source_time_ms)")
    reference_keys = list(zip(reference_id, reference_time))
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError("duplicate (reference_record_id, reference_time_ms)")
    result = np.array([source_rows.get(key, -1) for key in reference_keys], dtype=int)
    if require_all and np.any(result < 0):
        missing = reference_keys[int(np.flatnonzero(result < 0)[0])]
        raise ValueError(f"missing source record/time pair: {missing}")
    return result


def correct_clock(raw_time_ms, anchor_raw_ms, anchor_utc_ms):
    """Apply a per-source affine clock correction from NTP/PPS anchors."""
    raw = _strict_times(raw_time_ms, "raw_time_ms")
    anchor_raw = _strict_times(anchor_raw_ms, "anchor_raw_ms")
    anchor_utc = _strict_times(anchor_utc_ms, "anchor_utc_ms")
    if len(anchor_raw) != len(anchor_utc) or len(anchor_raw) < 2:
        raise ValueError("clock correction requires at least two matching clock anchors")
    if len(raw) and (raw[0] < anchor_raw[0] or raw[-1] > anchor_raw[-1]):
        raise ValueError("clock anchors must bracket raw_time_ms")
    slope, offset = np.polyfit(anchor_raw - anchor_raw[0], anchor_utc - anchor_utc[0], 1)
    corrected = anchor_utc[0] + np.rint((raw - anchor_raw[0]) * slope + offset).astype(np.int64)
    return _strict_times(corrected, "corrected_time_ms")


def nearest_indices(grid_time_ms, source_time_ms, tolerance_ms):
    """Offline nearest-sample indices, with earlier sample preferred on ties."""
    grid = _strict_times(grid_time_ms, "grid_time_ms")
    source = _strict_times(source_time_ms, "source_time_ms")
    if not np.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("tolerance_ms must be finite and non-negative")
    if not len(source):
        return np.full(len(grid), -1, dtype=int)
    right = np.searchsorted(source, grid).clip(0, len(source) - 1)
    left = np.maximum(np.searchsorted(source, grid) - 1, 0)
    row = np.where(abs(source[left] - grid) <= abs(source[right] - grid), left, right)
    return np.where(abs(source[row] - grid) <= tolerance_ms, row, -1)


@dataclass(frozen=True)
class ChannelAlignment:
    wb: np.ndarray
    nb: np.ndarray
    jitter_ms: np.ndarray
    valid: np.ndarray


def align_channels_offline(grid_time_ms, wb_time_ms, nb_time_ms, tolerance_ms):
    """Align each source to a 1 Hz record grid and apply the <20 ms jitter limit."""
    grid = _strict_times(grid_time_ms, "grid_time_ms")
    if np.any(np.diff(grid) != STEP_MS):
        raise ValueError("grid_time_ms must be a continuous 1 Hz grid")
    wb_time = _strict_times(wb_time_ms, "wb_time_ms")
    nb_time = _strict_times(nb_time_ms, "nb_time_ms")
    wb = nearest_indices(grid, wb_time, tolerance_ms)
    nb = nearest_indices(grid, nb_time, tolerance_ms)
    paired = (wb >= 0) & (nb >= 0)
    jitter = np.full(len(grid), np.nan)
    jitter[paired] = abs(wb_time[wb[paired]] - nb_time[nb[paired]])
    valid = paired & (jitter < CHANNEL_JITTER_LIMIT_MS)
    for index in (wb, nb):
        if len(np.unique(index[valid])) != valid.sum():
            raise ValueError("alignment would reuse a radiance frame across 1 Hz timestamps")
    return ChannelAlignment(wb, nb, jitter, valid)


def causal_indices(grid_time_ms, source_time_ms, max_age_ms):
    """Latest online input at or before each query, with an explicit age limit."""
    grid = _strict_times(grid_time_ms, "grid_time_ms")
    source = _strict_times(source_time_ms, "source_time_ms")
    if not np.isfinite(max_age_ms) or max_age_ms < 0:
        raise ValueError("max_age_ms must be finite and non-negative")
    if not len(source):
        return np.full(len(grid), -1, dtype=int)
    row = np.searchsorted(source, grid, side="right") - 1
    age = grid - source[np.maximum(row, 0)]
    return np.where((row >= 0) & (age <= max_age_ms), row, -1)


def outlier_mask(values, *, iqr_multiplier, combine):
    """Offline IQR and 3σ checks per column of a continuous record (A.1, F.3).

    The manuscript specifies both tests. Fence width and the Boolean combination
    are caller-supplied so their effect on the source quality mask is auditable.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError("outlier inputs must be [time,field] and mark missing values NaN")
    if not np.isfinite(iqr_multiplier) or iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be positive and finite")
    if combine not in {"either", "both"}:
        raise ValueError("combine must be either or both")
    mask = np.zeros(values.shape, dtype=bool)
    for column, field in enumerate(values.T):
        finite = np.isfinite(field)
        clean = field[finite]
        if len(clean) < 2:
            continue
        q1, q3 = np.percentile(clean, [25, 75])
        fence = iqr_multiplier * (q3 - q1)
        iqr = (clean < q1 - fence) | (clean > q3 + fence)
        sd = np.abs(clean - clean.mean()) > 3 * clean.std(ddof=1)
        mask[finite, column] = iqr | sd if combine == "either" else iqr & sd
    return mask


def contiguous_rows(time_ms, record_id):
    """Yield row indices of each record's continuous 1 Hz segments."""
    time_ms = utc_milliseconds(time_ms)
    record_id = _record_ids(record_id, len(time_ms), "record_id")
    groups = {}
    for row, record in enumerate(record_id):
        groups.setdefault(record, []).append(row)
    for record, rows in groups.items():
        rows = np.asarray(rows)
        times = _strict_times(time_ms[rows], f"time_ms in {record}")
        yield from np.split(rows, np.flatnonzero(np.diff(times) != STEP_MS) + 1)


def _gap_limit(max_gap_steps):
    if (not isinstance(max_gap_steps, (int, np.integer))
            or isinstance(max_gap_steps, (bool, np.bool_)) or max_gap_steps < 0):
        raise ValueError("max_gap_steps must be a non-negative integer")


def _repair_segments(values, segments, max_gap_steps, eligible, track_readiness):
    values = np.asarray(values)
    if values.ndim < 2 or values.dtype.kind != "f" or np.isinf(values).any():
        raise ValueError("values must be floating [time,...] and use NaN for missing fields")
    repaired = values.copy()
    if eligible is not None:
        eligible = np.asarray(eligible)
        if eligible.dtype != np.bool_:
            raise ValueError("eligible must be a Boolean array")
        eligible = np.broadcast_to(eligible, repaired.shape)
        repaired[~eligible] = np.nan
    ready = None
    if track_readiness:
        # A complete feature vector is known when its last field is known.
        shape = (len(repaired),) + (1,) * (repaired.ndim - 2)
        ready = np.broadcast_to(np.arange(len(repaired)).reshape(shape), repaired.shape[:-1]).copy()
    if not max_gap_steps or not len(repaired):
        return repaired, ready
    fields = repaired.reshape(len(repaired), -1)
    allowed = None if eligible is None else eligible.reshape(fields.shape)
    readiness = None if ready is None else ready.reshape(len(repaired), -1)
    for rows in segments:
        block = fields[rows]
        for column in np.flatnonzero(~np.isfinite(block).all(axis=0)):
            field = block[:, column]
            known = np.flatnonzero(np.isfinite(field))
            gaps = np.diff(known)
            for gap_index in np.flatnonzero((gaps > 1) & (gaps <= max_gap_steps + 1)):
                left, right = known[gap_index:gap_index + 2]
                gap = right - left - 1
                missing_rows = rows[left + 1:right]
                if allowed is not None and not allowed[missing_rows, column].all():
                    continue
                fraction = np.arange(1, gap + 1) / (gap + 1)
                field[left + 1:right] = field[left] + fraction * (field[right] - field[left])
                if readiness is not None:
                    vector = column // repaired.shape[-1]
                    readiness[missing_rows, vector] = np.maximum(
                        readiness[missing_rows, vector], rows[right],
                    )
        fields[rows] = block
    return repaired, ready


def interpolate_short_gaps(
    values, time_ms, record_id, max_gap_steps, *, eligible=None, return_ready_at=False,
):
    """Prepare bounded repairs with optional per-vector ready-at row indices.

    For a decision at row t, use a complete vector only when its ready-at row
    is at or before t. Input row indices increase within each record.
    """
    _gap_limit(max_gap_steps)
    values = np.asarray(values)
    if not values.ndim or len(values) != len(time_ms):
        raise ValueError("values must contain one row per timestamp")
    segments = tuple(contiguous_rows(time_ms, record_id))
    repaired, ready = _repair_segments(values, segments, max_gap_steps, eligible, return_ready_at)
    return (repaired, ready) if return_ready_at else repaired


def repair_causal_window(history, max_gap_steps, window_steps=8):
    """Repair the final window using a continuous record prefix ending at t.

    An earlier endpoint may precede the window. Unresolved fields remain NaN.
    Reset history at record and 1 Hz continuity boundaries before calling.
    """
    _gap_limit(max_gap_steps)
    if not isinstance(window_steps, int) or isinstance(window_steps, bool) or window_steps <= 0:
        raise ValueError("window_steps must be a positive integer")
    history = np.asarray(history)
    if history.ndim != 3:
        raise ValueError("history must have shape [time,nozzle,feature]")
    history = history[-(window_steps + max_gap_steps):]
    repaired, _ = _repair_segments(history, (np.arange(len(history)),), max_gap_steps, None, False)
    return repaired[-window_steps:]
