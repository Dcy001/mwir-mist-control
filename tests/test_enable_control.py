import json

import numpy as np
import pytest

from mwir_control.control import select_action
from mwir_control.cycle import ActualReceipt, CycleController, CycleKey, EnablePlan, Observation, cycle_log
from mwir_control.nozzles import Action, Nozzle, NozzleRegistry


STEP = 1_000_000_000


def _registry(*regions):
    return NozzleRegistry([
        Nozzle(
            f"N{i}", region, "SC" if region.startswith("SC_") else "SCC-port",
            [0, 1, 2], np.eye(3), [1, 0, 0],
            0.30 if region.startswith("SC_") else 0.80,
        )
        for i, region in enumerate(regions)
    ])


def _ingest(controller, second, actual, *, record="trial"):
    key = CycleKey(record, second * STEP)
    return controller.ingest(
        Observation(key, np.ones((2, 2, 2)), np.ones((2, 2), dtype=bool), second, "camera"),
        ActualReceipt(key, controller.registry.ids, actual, "measured"),
    )


def _enable(controller, feedback, enabled, *, offset=0, source="authorised schedule"):
    return EnablePlan(
        feedback.key.successor(), controller.registry.ids, enabled, source, feedback.key.timestamp_ns + offset,
    )


def test_projection_requires_explicit_mask_and_preserves_no_mist():
    registry = _registry("SC_1", "SC_2")
    raw = np.array([[0.30, -5, 35], [-3, -8, -90]])
    with pytest.raises(TypeError):
        registry.project(raw)
    with pytest.raises(ValueError, match="Boolean"):
        registry.project(raw, [1, 0])
    with pytest.raises(ValueError, match="required"):
        registry.project(None, [True, False])
    all_on = registry.project(raw, [True, True])
    np.testing.assert_equal(all_on.values, [[0.30, 0.40, 8], [0.137, 0.40, -8]])
    mixed = registry.project(raw, [True, False])
    np.testing.assert_equal(mixed.values, [[0.30, 0.40, 8], [0, 0, 0]])
    assert registry.flows(mixed)[1] == 0.0
    all_off = registry.project(None, [False, False])
    np.testing.assert_equal(all_off.values, np.zeros((2, 3)))
    np.testing.assert_equal(registry.flows(all_off), np.zeros(2))


def test_scc_group_enable_validation_precedes_hydraulics():
    registry = _registry("SCC-F", "SCC-F")
    values = [[0.92, 0.70, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match="SCC-F requires all"):
        registry.project(values, [True, False])
    with pytest.raises(ValueError, match="SCC-F requires all"):
        registry.flows(Action(values, [True, False]))
    off = registry.project(None, [False, False])
    np.testing.assert_equal(registry.flows(off), [0, 0])


def test_vso_never_searches_or_activates_a_disabled_region():
    registry = _registry("SC_1", "SC_2")
    evaluated = []

    def assess(action):
        evaluated.append(action)
        return {"SC_1": 20 * action.values[0, 1] - 14, "SC_2": 100.0}

    result = select_action(
        [[0.30, 0.90, 0.0], [np.nan, np.nan, np.nan]], registry, assess, enabled=[True, False],
    )
    assert [search.nozzle_id for search in result.searches] == ["N0"]
    assert result.final_scores["SC_2"] == 100
    for action in evaluated:
        np.testing.assert_equal(action.enabled, [True, False])
        np.testing.assert_equal(action.values[1], [0, 0, 0])


def test_all_off_cycle_can_issue_at_start_and_requires_matched_causal_plan():
    registry = _registry("SC_1", "SC_2")
    evaluations = []

    def model(action, context):
        evaluations.append((action, context))
        return np.zeros((2, 2, 2))

    def must_not_predict(windows):
        pytest.fail("An all-off target never needs LSTM inference")

    def score(frames, context, action):
        assert not action.enabled.any()
        return {}

    controller = CycleController(
        registry, model, lambda *args: np.full((2, 25), np.nan), must_not_predict, score,
    )
    actual = Action([[0.30, 0.5, 0], [0.30, 0.5, 0]], [True, True])
    feedback = _ingest(controller, 0, actual)
    assert controller.propose(None) is None
    assert controller.proposal_status == "missing_enable_plan"
    with pytest.raises(ValueError, match="unavailable"):
        controller.propose(_enable(controller, feedback, [False, False], offset=1))
    incorrect = EnablePlan(CycleKey("other", STEP), registry.ids, [False, False], "schedule", 0)
    with pytest.raises(ValueError, match="next cycle"):
        controller.propose(incorrect)
    unordered = EnablePlan(CycleKey("trial", STEP), registry.ids[::-1], [False, False], "schedule", 0)
    with pytest.raises(ValueError, match="ordering"):
        controller.propose(unordered)

    requested = _enable(controller, feedback, [False, False])
    plan = controller.propose(requested)
    assert controller.proposal_status == "planned"
    assert plan.selection.raw is None
    assert plan.inference_ms == 0
    assert plan.selection.evaluations == 1
    assert not evaluations[1][0].enabled.any()
    assert evaluations[0][0] is actual
    assert controller.propose(requested) is plan
    with pytest.raises(ValueError, match="different enable plan"):
        controller.propose(_enable(controller, feedback, [True, True]))
    log = cycle_log(feedback, plan, requested_enable=requested, proposal_status=controller.proposal_status)
    assert log["plan"]["raw"] is None
    assert log["plan"]["final_enabled"] == [False, False]
    assert log["requested_enable"]["available_timestamp_ns"] == 0
    assert log["requested_enable"]["source"] == "authorised schedule"
    json.dumps(log, allow_nan=False)


def test_reenable_uses_available_disabled_history_and_causal_feature_repair():
    registry = _registry("SC_1", "SC_2")
    predictions = []

    def features(observed, modelled, actual, second):
        assert not actual.enabled.any()
        result = np.full((2, 25), np.nan)
        result[0] = second
        if second in (1, 7):
            result[0, 0] = np.nan
        return result

    def predict(windows):
        predictions.append(windows)
        return np.array([[0.30, 0.50, 0]])

    def score(frames, context, action):
        assert action.enabled.tolist() == [True, False]
        return {"SC_1": 0}

    controller = CycleController(
        registry, lambda *args: np.zeros((2, 2, 2)), features, predict, score, max_gap_steps=1,
    )
    actual = Action.off(2)
    for t in range(8):
        feedback = _ingest(controller, t, actual)
    request = _enable(controller, feedback, [True, False])
    assert controller.propose(request) is None
    assert controller.proposal_status == "invalid_enabled_windows"
    assert not predictions

    feedback = _ingest(controller, 8, actual)
    request = _enable(controller, feedback, [True, False])
    plan = controller.propose(request)
    assert len(predictions) == 1
    assert predictions[0].shape == (1, 8, 25)
    np.testing.assert_equal(predictions[0][0, :, 0], np.arange(1, 9))
    np.testing.assert_equal(plan.selection.final.enabled, [True, False])
    log = cycle_log(feedback, plan, requested_enable=request)
    assert log["plan"]["raw"] == [[0.30, 0.50, 0], None]
    np.testing.assert_allclose(feedback.residual, 1)
