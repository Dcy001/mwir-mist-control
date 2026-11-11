import json

import numpy as np
import pytest

from mwir_control.control import LocalCandidate, final_choice, finite_difference, select_action, visibility
from mwir_control.cycle import (
    ActualReceipt, CommandAck, CycleController, CycleKey, EnablePlan, Observation, cycle_log, replay,
)
from mwir_control.evaluation_io import evaluate_manifest
from mwir_control.metrics import (
    LATENCY_STAGES, STUDY_ROI_IDS, Trial, compare_trials, latency_metrics, prediction_metrics, water_metrics,
)
from mwir_control.nozzles import (
    Action, Nozzle, NozzleRegistry, SCC_A_TOTAL, SCC_F_TOTAL, SC_REGION_IDS, published_flow_curve,
)


def nozzle(name="N", region="SC1", kind="SC", flow_extension=()):
    return Nozzle(name, region, kind, [0, 1, 2], np.eye(3), [1, 0, 0], 0.30, flow_extension)


def test_flow_group_reconstruction_and_off_state():
    _, ps = published_flow_curve("SCC-port")
    _, m = published_flow_curve("SCC-midships")
    np.testing.assert_allclose(2 * ps + m, SCC_A_TOTAL)
    np.testing.assert_allclose(4 * ps + m, SCC_F_TOTAL)
    single = nozzle()
    assert single.flow_l_min(0.36, 0.5) == pytest.approx(1.6125)
    with pytest.raises(ValueError, match="flow_extension"):
        single.flow_l_min(0.15, 0.5)
    extended = nozzle(flow_extension=((0.137, 2.2),))
    assert extended.flow_l_min(0.137, 0.5) == pytest.approx(1.1)
    registry = NozzleRegistry([single])
    np.testing.assert_equal(registry.flows(Action.off(1)), [0.0])
    projected = registry.project([[-1, -1, 90]], [True])
    np.testing.assert_equal(projected.values, [[0.137, 0.4, 8.0]])
    with pytest.raises(ValueError, match="Disabled"):
        Action([[0.3, 0.2, 0]], [False])


def test_visibility_uses_channel_specific_noise_sample_std_and_glint():
    masks = {
        "Sea": np.array([[True, True, False]]),
        "SC1": np.array([[False, False, True]]),
    }
    radiance = np.array([[[1.0, 3.0, 8.0]], [[1.0, 1.0, 6.0]]])
    valid = np.ones((1, 3), dtype=bool)
    expected = 0.75 * 6 / np.sqrt(2) + 0.25 * 5 / 2
    assert visibility(radiance, masks, valid, [0.5, 2.0], 0.25)["SC1"] == pytest.approx(expected)
    assert visibility(radiance, masks, valid, [0.5, 2.0], 1)["SC1"] == pytest.approx(2.5)
    radiance[:, :, -1] = 0.5
    assert visibility(radiance, masks, valid, [0.5, 2.0], 0.25)["SC1"] == 0


def test_search_uses_normalised_coordinates_and_early_exit():
    np.testing.assert_allclose(
        finite_difference(lambda p: 2 * p[0] + 5 * p[1] - p[2], [0, 0.5, 1]),
        [2, 5, -1],
    )
    registry = NozzleRegistry([nozzle(flow_extension=((0.137, 2.2),))])
    calls = []

    def evaluator(action):
        calls.append(action)
        duty = action.values[0, 1]
        return {"SC1": 20 * duty - 14}

    raw = [[0.3, 0.9, 0.0]]
    result = select_action(raw, registry, evaluator, enabled=[True])
    search = result.searches[0]
    assert result.projected_scores["SC1"] == pytest.approx(4.0)
    assert result.final_scores["SC1"] <= 3.0
    assert search.iterations == 1
    assert search.reason == "threshold_met"
    assert len(search.probes) == 6
    assert result.evaluations == len(calls)
    assert result.final.values[0, 1] < result.projected.values[0, 1]
    np.testing.assert_equal(result.raw, raw)


def test_each_vso_search_is_against_the_projected_regional_set():
    rows = [
        Nozzle(f"N{i}", "SCC-F", "SCC-port", [i, 0, 1], np.eye(3), [1, 0, 0], 0.827)
        for i in range(2)
    ]
    registry = NozzleRegistry(rows)
    calls = []

    def evaluator(action):
        calls.append(action.values.copy())
        return {"SCC-F": 15 * action.values[:, 1].sum() - 23}

    result = select_action([[0.8, 0.9, 0], [0.8, 0.9, 0]], registry, evaluator, enabled=[True, True])
    assert len(result.searches) == 2
    assert all(np.sum(np.any(values != result.projected.values, axis=1)) <= 1 for values in calls[:-1])
    assert np.all(result.final.values[:, 1] < result.projected.values[:, 1])
    np.testing.assert_allclose(calls[-1], result.final.values)


def test_equation_27_threshold_then_tolerance():
    def candidate(x, score):
        return LocalCandidate((x, 0, 0), (x, 0, 0), score, score)

    initial = candidate(0, 5)
    close = candidate(0.05, 4)
    far = candidate(0.051, 4)
    meeting = candidate(0.01, 3)
    assert final_choice(initial, close)[0] is initial
    assert final_choice(initial, far)[0] is far
    assert final_choice(initial, meeting)[0] is meeting


def test_replay_residual_uses_recorded_actual_action():
    registry = NozzleRegistry([nozzle()])
    evaluated = []

    def evaluator(action, context):
        evaluated.append((action.values[0, 0], context))
        return np.full((2, 1, 2), action.values[0, 0])

    def features(observed, modelled, action, context):
        return np.full((1, 25), (observed - modelled).mean())

    def predict(windows):
        assert windows.shape == (1, 8, 25)
        assert windows[0, 0, 0] == pytest.approx(1.7)
        return [[0.31, 0.5, 0]]

    controller = CycleController(registry, evaluator, features, predict, lambda f, c, action: {"SC1": 0})
    actual = Action([[0.30, 0.5, 0]], [True])
    entries = []
    plans = {}
    for i in range(10):
        key = CycleKey("r", i * 10 ** 9)
        entries.append((
            Observation(key, np.full((2, 1, 2), 2.0), np.ones((1, 2), dtype=bool), i, "camera", 2.0),
            ActualReceipt(key, registry.ids, actual, "logged action"),
        ))
        target = key.successor()
        plans[target] = EnablePlan(target, registry.ids, [True], "plan record", key.timestamp_ns)
    result = list(replay(controller, entries, plans))
    assert all(plan is None for _, plan in result[:7])
    assert result[7][1].target == entries[8][0].key
    assert evaluated[7:10] == [(0.30, 7), (0.31, 7), (0.30, 8)]
    np.testing.assert_allclose(result[8][0].residual, 1.7)
    with pytest.raises(ValueError, match="same record and timestamp"):
        controller.ingest(entries[-1][0], entries[-2][1])

    class TestActuator:
        def issue(self, plan):
            return CommandAck(plan.target, True, "ack-1")

    feedback, plan = result[-1]
    dispatch = controller.issue(plan, TestActuator())
    log = cycle_log(feedback, plan, dispatch)
    assert set(log["latency_stages_ms"]) == set(LATENCY_STAGES)
    assert log["end_to_end_ms"] >= 2.0
    json.dumps(log, allow_nan=False)
    with pytest.raises(ValueError, match="already attempted"):
        controller.issue(plan, TestActuator())
    key = CycleKey("r", 11 * 10 ** 9)
    controller.ingest(
        Observation(key, entries[-1][0].frames, entries[-1][0].valid, 11, "camera"),
        ActualReceipt(key, registry.ids, actual, "logged action"),
    )
    assert controller.propose(EnablePlan(key.successor(), registry.ids, [True], "plan record", key.timestamp_ns)) is None


def test_prediction_water_and_latency_metrics():
    y = np.array([[[1.0, 0, 0]], [[2.0, 0, 0]]])
    metrics = prediction_metrics(y + 0.25, y, [[0, -1, -1]], [[2, 1, 1]])
    np.testing.assert_allclose(metrics["rmse"], 0.25)
    assert metrics["r2"][0, 0] == pytest.approx(0.75)
    assert np.isnan(metrics["r2"][0, 1])
    np.testing.assert_equal(metrics["cvr_pct"], [[50, 0, 0]])

    flow = np.zeros((60, 286))
    flow[:, 0] = 60.0
    capacity = np.full(286, 60.0)
    usage = water_metrics(flow, capacity)
    assert usage["water_m3"] == pytest.approx(0.06)
    assert usage["full_load_24h_m3"] == pytest.approx(24710.4)
    assert usage["saving_pct"] == pytest.approx(100 * (1 - 0.06 / 24710.4))

    stages = {name: np.zeros(3) for name in LATENCY_STAGES}
    stages["phys"] = np.array([999, 1000, 1001])
    latency = latency_metrics(stages)
    assert latency["deadline_miss_pct"] == pytest.approx(100 / 3)
    assert latency["jitter_ms"] == pytest.approx(np.sqrt(2 / 3))


def test_prediction_metrics_exclude_disabled_targets_from_error_and_cvr():
    labels = np.array([[[0.3, 0.5, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    predicted = np.array([[[0.32, 0.5, 0], [np.nan, np.nan, np.nan]], [[0, 0, 0], [0, 0, 0]]])
    lower = [[0.2, 0.4, -8]] * 2
    upper = [[0.35, 1, 8]] * 2
    enabled = [[True, False], [False, False]]
    result = prediction_metrics(predicted, labels, lower, upper, target_enabled=enabled)
    np.testing.assert_equal(result["evaluated_count"], [1, 0])
    np.testing.assert_allclose(result["mae"][0], [0.02, 0, 0])
    np.testing.assert_allclose(result["cvr_pct"][0], [0, 0, 0])
    assert np.isnan(result["rmse"][1]).all()
    assert np.isnan(result["r2"]).all()
    with pytest.raises(ValueError, match="finite for every enabled target"):
        prediction_metrics(predicted, labels, lower, upper, target_enabled=[[True, True], [False, False]])


@pytest.fixture
def comparison_inputs():
    times = np.array([7 * 3600, 12 * 3600, 22 * 3600], dtype=np.int64) * 10 ** 9
    masks = {
        "Sea": np.array([[True, True, False, False, False, False]]),
        "SCC-F": np.array([[False, False, True, False, False, False]]),
        "SCC-A": np.array([[False, False, False, True, False, False]]),
        "SC_1": np.array([[False, False, False, False, True, True]]),
        "Global": np.array([[False, False, True, True, True, True]]),
    }
    base = np.broadcast_to([1, 1, 5, 5, 8, 10], (3, 2, 1, 6)).copy()
    strategy = np.broadcast_to([1, 1, 3, 3, 4, 5], base.shape).copy()
    all_valid = np.ones((3, 1, 6), dtype=bool)
    some_valid = all_valid.copy()
    some_valid[:, 0, 5] = False
    base_trial = Trial(times, "grid", base, all_valid, base[:, 0] + 20)
    treatment = Trial(times, "grid", strategy, some_valid, strategy[:, 0] + 20)
    kwargs = dict(
        masks=masks, pixel_area_m2=1, noise=[1, 1], glint=np.zeros(3), day_start_ns=0, epsilon=1e-8,
    )
    return base_trial, treatment, kwargs


def test_strategy_metrics_use_shared_valid_pixels_and_aligned_baseline(comparison_inputs):
    base_trial, treatment, kwargs = comparison_inputs
    kwargs["roi_scope"] = {"mode": "subset", "subset_id": "test", "roi_ids": list(kwargs["masks"])}
    result = compare_trials(base_trial, {"adaptive": treatment}, **kwargs)["adaptive"]
    assert result["main_channel"]["SCC-F"]["timestamps"] == 1
    assert result["main_channel"]["SC_1"]["timestamps"] == 3
    assert result["main_channel"]["SC_1"]["baseline_radiance"] == 8
    assert result["main_channel"]["SC_1"]["attenuation_pct"] == 50
    assert result["main_channel"]["SC_1"]["temperature_reduction_c"] == 4.5
    assert result["auxiliary_channel"]["SC_1"]["top_1pct_radiance"] == 4
    assert result["visibility"]["compression_pct"] == pytest.approx(100 * (1 - 3 / 7))
    bad = Trial(treatment.timestamps_ns + 1, "grid", treatment.radiance, treatment.valid)
    with pytest.raises(ValueError, match="share timestamps"):
        compare_trials(base_trial, {"adaptive": bad}, **kwargs)


@pytest.fixture
def full_comparison_inputs(comparison_inputs):
    _, _, kwargs = comparison_inputs
    masks = kwargs["masks"]
    masks.update({region: masks["SC_1"].copy() for region in SC_REGION_IDS})
    return comparison_inputs


def test_full_comparison_requires_and_reports_entire_study_roi_list(full_comparison_inputs):
    baseline, treatment, kwargs = full_comparison_inputs
    result = compare_trials(baseline, {"adaptive": treatment}, **kwargs)["adaptive"]
    assert result["roi_scope"]["mode"] == "full"
    assert tuple(result["roi_scope"]["roi_ids"]) == STUDY_ROI_IDS
    assert result["roi_scope"]["coverage"]["evaluated_spray_rois"] == 280
    assert set(result["auxiliary_channel"]) == set(SC_REGION_IDS)
    np.testing.assert_equal(result["visibility"]["baseline_max"], [7, 7, 7])
    assert "baseline_subset_max" not in result["visibility"]


def test_full_comparison_reports_missing_and_unknown_roi_names(full_comparison_inputs):
    baseline, treatment, kwargs = full_comparison_inputs
    masks = kwargs["masks"]
    masks.pop("SC_278")
    with pytest.raises(ValueError, match="missing ROI masks: SC_278"):
        compare_trials(baseline, {"adaptive": treatment}, **kwargs)
    masks["SC_278"] = masks["SC_1"].copy()
    masks["SC_279"] = masks["SC_1"].copy()
    with pytest.raises(ValueError, match="unknown ROI names: SC_279"):
        compare_trials(baseline, {"adaptive": treatment}, **kwargs)


def test_comparison_reports_missing_shared_support_with_roi_and_timestamp(full_comparison_inputs):
    baseline, treatment, kwargs = full_comparison_inputs
    masks = kwargs["masks"]
    masks["SC_278"] = np.array([[False, False, False, False, False, True]])
    timestamp = baseline.timestamps_ns[0]
    with pytest.raises(ValueError, match=rf"timestamp_ns={timestamp}.*valid pixels.*ROI SC_278"):
        compare_trials(baseline, {"adaptive": treatment}, **kwargs)


def test_evaluation_manifest_uses_explicit_roi_subset(comparison_inputs, tmp_path):
    baseline, treatment, kwargs = comparison_inputs
    masks = {name: kwargs["masks"][name] for name in ("Sea", "SC_1")}
    for name, trial in (("base", baseline), ("treatment", treatment)):
        np.savez(
            tmp_path / f"{name}.npz", grid_id="grid", source=name, channels=["3-5", "3.9"],
            radiance_units="W m-2 sr-1", timestamps_ns=trial.timestamps_ns,
            radiance=trial.radiance, valid=trial.valid,
        )
    np.savez(
        tmp_path / "support.npz", grid_id="grid", timestamps_ns=baseline.timestamps_ns,
        roi_ids=list(masks), roi_masks=list(masks.values()), noise=kwargs["noise"],
        glint=kwargs["glint"], pixel_area_m2=kwargs["pixel_area_m2"],
    )
    spec = {
        "no_mist": "base.npz", "strategies": {"adaptive": "treatment.npz"}, "support_npz": "support.npz",
        "day_start_ns": kwargs["day_start_ns"], "epsilon": kwargs["epsilon"],
    }
    manifest = tmp_path / "metrics.json"
    manifest.write_text(json.dumps({"schema_version": 1, "comparison": spec}), encoding="utf-8")
    with pytest.raises(ValueError, match="Full comparison ROI scope: missing ROI masks"):
        evaluate_manifest(manifest)
    spec["roi_scope"] = {"mode": "subset", "subset_id": "side-camera", "roi_ids": list(masks)}
    manifest.write_text(json.dumps({"schema_version": 1, "comparison": spec}), encoding="utf-8")
    result = evaluate_manifest(manifest)["comparison"]["adaptive"]
    assert result["roi_scope"] == {
        "mode": "subset", "subset_id": "side-camera", "roi_ids": ["Sea", "SC_1"],
        "coverage": {"evaluated_spray_rois": 1, "study_spray_rois": 280, "evaluated_sc_rois": 1, "study_sc_rois": 278},
    }
    assert set(result["main_channel"]) == {"SC_1"}
    assert "baseline_max" not in result["visibility"]
    np.testing.assert_equal(result["visibility"]["baseline_subset_max"], [7, 7, 7])
    np.testing.assert_equal(result["visibility"]["strategy_subset_max"], [3, 3, 3])


def test_registry_checks_hardware_inventory_and_right_handed_side():
    def row(index, region, kind, lateral, reference):
        return {
            "nozzle_id": f"N{index}", "region": region, "kind": kind,
            "position_m": [float(index), lateral, 0.0], "mount_rotation": np.eye(3),
            "deflection_axis": [1, 0, 0], "reference_pressure_mpa": reference,
        }

    rows = [
        row(i, f"SC_{i + 1}", "SC", 5.0 if i < 139 else -5.0, 0.30)
        for i in range(278)
    ]
    for group, per_side in (("SCC-F", 2), ("SCC-A", 1)):
        for kind in ["SCC-port"] * per_side + ["SCC-starboard"] * per_side + ["SCC-midships"]:
            rows.append(row(len(rows), group, kind, 0.0, 0.827))
    registry = NozzleRegistry.from_rows(rows)
    assert len(registry) == 286
    assert registry.region_ids == tuple(f"SC_{i}" for i in range(1, 279)) + ("SCC-F", "SCC-A")
    rows[0]["position_m"][1] *= -1
    with pytest.raises(ValueError, match="139 port"):
        NozzleRegistry.from_rows(rows)
    rows[0]["region"] = "Sea"
    with pytest.raises(ValueError, match="invalid spray region"):
        NozzleRegistry.from_rows(rows)
