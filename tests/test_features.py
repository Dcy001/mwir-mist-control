"""Regression checks for the pixel-domain data path."""
import numpy as np
import pytest

from mwir_control.data import (
    align_channels_offline, causal_indices, correct_clock, interpolate_short_gaps,
    keyed_indices, outlier_mask, repair_causal_window,
)
from mwir_control.features import FEATURE_NAMES, FEATURE_UNITS, build_features
from mwir_control.roi import build_masks, glint_risk, hotspot_statistics, roi_statistics


def test_projected_roi_shapes():
    fwd = np.array([[15, 8], [17, 8], [17, 11], [15, 11]])
    aft = fwd + [20, 0]
    masks = build_masks(
        (55, 60), {"SCC-F": fwd, "SCC-A": aft},
        {"SC_1": [[10, 16], [45, 16]]}, [[4, 30], [50, 30]],
    )
    assert masks["SCC-F"][8, 9] and not masks["SCC-F"][8, 8]
    assert not masks["SCC-F"][8, 35] and masks["SCC-A"][8, 35]
    assert masks["SC_1"][24, 30] and not masks["SC_1"][25, 30]
    assert not masks["SC_1"][16, 9] and not masks["SC_1"][16, 46]
    assert masks["Sea"].sum() == 47 * 20
    assert masks["Sea"][30, 10] and not masks["Sea"][50, 10]


def test_statistics_and_hotspots_with_missing_pixels():
    field = np.full((16, 20), -1.0)
    source = np.zeros_like(field, dtype=bool)
    source[2:15, 1:] = True
    sea = np.zeros_like(source)
    sea[0, 0:2] = True
    field[sea] = 0
    valid = np.ones_like(source)
    field[3:6, 4:7] = 1
    field[4, 5] = -1   # Cross-kernel closing fills the centre.
    field[6:8, 7:9] = 1  # 8-connectivity links the diagonal component.
    field[12:14, 14:16] = 1  # Four pixels are below the minimum.
    hot = hotspot_statistics(field, source, sea, valid, 0.02)
    assert hot.count == 1
    assert hot.area_m2 == pytest.approx(hot.mask.sum() * 0.02)
    assert hot.mask[4, 5] and hot.mask[7, 8] and not hot.mask[12, 14]
    valid[4, 5] = False
    assert not hotspot_statistics(field, source, sea, valid, 0.02).mask[4, 5]
    field[0, 1] = np.nan
    stats = roi_statistics(field, sea, valid)
    assert stats.count == 1 and stats.mean == 0 and np.isnan(stats.std)
    assert np.isnan(hotspot_statistics(field, source, sea, valid, 0.02).area_m2)
    stats = roi_statistics([[1, 3, np.nan]], [[True] * 3], [[True] * 3])
    assert stats.std == pytest.approx(np.sqrt(2))


def test_radiance_residual_and_feature_order():
    sea = np.zeros((7, 7), dtype=bool)
    sea[0] = True
    sc = np.zeros_like(sea)
    sc[2:5, 2:5] = True
    stack = np.zeros_like(sea)
    stack[5:, 2:5] = True
    modelled = np.stack([np.full(sea.shape, 2.), np.full(sea.shape, 5.)])
    observed = modelled.copy()
    observed[0, sc] += 3
    observed[1, sc] += 1
    observed[0, stack] += 4
    observed[1, stack] += 2
    actual = np.array([[0.20, 0.5, 4], [0.25, 0.6, 3], [0.8, 0.9, -2]])
    features = build_features(
        observed, modelled, np.ones_like(sea), {"Sea": sea, "SC_1": sc, "SCC-A": stack},
        0.1, ["SC_1", "SC_1", "SCC-A"], actual, [15, 10000, 4, 18], [5, 90, 3, -1], 0.3,
    )
    assert features.shape == (3, 25)
    assert len(FEATURE_NAMES) == len(FEATURE_UNITS) == 25
    assert np.allclose(features[0, :13], [15, 10000, 4, 18, 5, 1, 0, 3, -1, .2, .5, 4, .3])
    assert np.allclose(features[0, 13:], [5, 3, 0, 1, .9, 2, 6, 1, 0, 1, .9, 5])
    assert np.allclose(features[0, 13:], features[1, 13:])
    assert features[2, 14] == 4


def test_glint_geometry():
    rays = np.array([[[0, 0, -1.], [0, -1., 0]]])
    assert glint_risk(rays, [[True, False]], 90, 0, 0) == pytest.approx(1)
    assert glint_risk(rays, [[False, True]], 90, 0, 0) < 1e-8


def test_exact_pairs_clock_and_cross_channel_jitter():
    assert np.array_equal(
        keyed_indices([1000, 2000], ["run", "run"], [2000, 1000], ["run", "run"]), [1, 0],
    )
    with pytest.raises(ValueError, match="missing source record"):
        keyed_indices([1000], ["run"], [1000], ["other"])
    corrected = correct_clock([1100, 2100], [100, 3100], [0, 3003])
    assert np.array_equal(corrected, [1001, 2002])
    matched = align_channels_offline([1000, 2000], [998, 2005], [1002, 1985], 50)
    assert np.array_equal(matched.jitter_ms, [4, 20])
    assert np.array_equal(matched.valid, [True, False])
    assert np.array_equal(causal_indices([1000, 2000], [1002, 1999], 100), [-1, 1])


def test_offline_quality_and_interpolation_respect_boundaries():
    values = np.r_[np.zeros(20), 20.][:, None]
    assert outlier_mask(values, iqr_multiplier=1.5, combine="both")[-1, 0]
    time_ms = [0, 1000, 2000, 3000, 5000, 6000, 7000]
    record_id = ["x", "x", "x", "y", "y", "y", "y"]
    raw = np.array([[1.], [np.nan], [3.], [np.nan], [5.], [np.nan], [7.]])
    filled = interpolate_short_gaps(raw, time_ms, record_id, 1)
    assert np.allclose(filled[:, 0], [1, 2, 3, np.nan, 5, 6, 7], equal_nan=True)
    eligible = np.ones_like(raw, dtype=bool)
    eligible[1] = False
    filled = interpolate_short_gaps(raw, time_ms, record_id, 1, eligible=eligible)
    assert np.isnan(filled[1, 0])


def test_bounded_repair_shares_online_cutoff_and_offline_readiness():
    history = np.arange(10, dtype=float).reshape(10, 1, 1)
    history[1] = np.nan  # Left endpoint precedes the eventual eight-step window.
    history[8] = np.nan  # Missing terminal field as of t=8.
    repaired, ready = interpolate_short_gaps(
        history, np.arange(10) * 1000, ["r"] * 10, 1, return_ready_at=True,
    )
    assert ready[1, 0] == 2 and ready[8, 0] == 9
    observed_at_8 = repair_causal_window(history[:9], 1)
    assert observed_at_8[0, 0, 0] == 1 and np.isnan(observed_at_8[-1, 0, 0])
    offline_at_8 = np.where(ready[1:9, :, None] <= 8, repaired[1:9], np.nan)
    assert np.allclose(offline_at_8, observed_at_8, equal_nan=True)
    assert repair_causal_window(history, 1)[-2, 0, 0] == 8
    history[9] = 100
    repaired_later, ready_later = interpolate_short_gaps(
        history, np.arange(10) * 1000, ["r"] * 10, 1, return_ready_at=True,
    )
    offline_at_8 = np.where(ready_later[1:9, :, None] <= 8, repaired_later[1:9], np.nan)
    assert np.allclose(offline_at_8, observed_at_8, equal_nan=True)
