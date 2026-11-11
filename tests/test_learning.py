"""Appendix E/F ordering, actual training step and portable checkpoint."""
import numpy as np
import pytest
import torch

from mwir_control.data import interpolate_short_gaps
from mwir_control.features import ACTION_UNITS, FEATURE_NAMES, FEATURE_UNITS
from mwir_control.learning import (
    Normalisation, SprayLSTM, TrainingRecords, chronological_split, index_windows,
    learning_rate, load_predictor, load_training_npz, train, training_loss,
)


def test_window_targets_split_and_predecessors():
    t = np.arange(120, dtype=np.int64) * 1000
    x = np.zeros((120, 2, 25))
    y = np.zeros((120, 2, 3))
    visibility = np.ones((120, 2))
    present = np.ones((120, 2), dtype=bool)
    record = np.where(np.arange(120) < 48, "A", "B")
    subsets = chronological_split(t)
    assert tuple(map(len, subsets)) == (96, 12, 12)
    first = subsets[0]
    x[9, 1] = np.nan
    index = index_windows(t[first], record[first], x[first], y[first], visibility[first], present[first])
    assert np.array_equal(index.row_windows[0], np.arange(8))
    assert index.row_targets[0] == 8
    assert index.row_windows[40, 0] == 48
    assert index.row_targets[40] == 56
    sample_time, nozzle = index.sample_windows, index.sample_nozzles
    assert not index.previous_valid[(sample_time == 40) & (nozzle == 0)].item()
    assert index.previous_valid[(sample_time == 41) & (nozzle == 1)].item()
    assert not index.previous_valid[(sample_time == 10) & (nozzle == 1)].item()


def test_normalisation_and_schedule():
    transform = Normalisation.fit(np.array([[1., 5], [3, 5]]))
    assert np.allclose(transform.mean, [2, 5])
    assert np.allclose(transform.scale, [1, 1])
    assert np.allclose(transform.inverse(transform.transform(np.array([4., 5]))), [4, 5])
    assert learning_rate(1, 180, 0) == pytest.approx(0.0002)
    assert learning_rate(5, 180, 0) == pytest.approx(0.001)
    assert learning_rate(180, 180, 0) == pytest.approx(0.00001)


def test_source_stratified_chronological_split():
    times = np.arange(200, dtype=np.int64) * 1000
    sources = np.repeat(["field", "simulation"], 100)
    global_split = chronological_split(times)
    assert set(sources[global_split[-1]]) == {"simulation"}
    train_rows, validation_rows, test_rows = chronological_split(times, sources)
    assert np.array_equal(validation_rows, np.r_[80:90, 180:190])
    assert np.array_equal(test_rows, np.r_[90:100, 190:200])
    assert len(train_rows) == 160


def test_model_layers_and_weighted_loss():
    lstm = SprayLSTM().eval()
    assert [layer.hidden_size for layer in lstm.layers] == [128, 64, 32]
    assert lstm(torch.zeros((3, 8, 25))).shape == (3, 3)
    prediction = torch.tensor([[2., 2, 2], [1, 1, 1]])
    prev = torch.zeros_like(prediction)
    result = training_loss(
        prediction, prev, prev, torch.tensor([True, False]),
        torch.zeros(2, 3), torch.ones(2, 3), torch.zeros(3), torch.full((3,), 2.), torch.tensor([1.5, 1.]),
    )
    assert result.item() == pytest.approx((1.5 * (4 + .1 * 27 + .01 * 48) + (1 + .1 * 3)) / 2)


def test_future_data_cannot_fill_terminal_gap_or_enable_smoothing():
    times = np.arange(14, dtype=np.int64) * 1000
    records = np.repeat("r", len(times))
    x = np.ones((len(times), 1, 25))
    x[7, 0, 0] = np.nan
    y = np.zeros((len(times), 1, 3))
    scores = np.ones((len(times), 1))
    present = np.ones((len(times), 1), dtype=bool)
    working = present.copy()
    working[9, 0] = False
    for later_value in (2., 70.):
        x[8, 0, 0] = later_value
        repaired, ready = interpolate_short_gaps(x, times, records, 1, return_ready_at=True)
        index = index_windows(
            times, records, repaired, y, scores, present, ready_at=ready, target_working=working,
        )
        target_rows = index.row_targets[index.sample_windows]
        assert np.array_equal(target_rows, [10, 11, 12, 13])
        assert np.array_equal(index.previous_valid, [False, True, True, True])
        assert index.filters["incomplete_causal_history"] == 1
    working[9, 0] = True
    index = index_windows(
        times, records, repaired, y, scores, present, ready_at=ready, target_working=working,
    )
    assert np.array_equal(index.row_targets[index.sample_windows], [9, 10, 11, 12, 13])
    assert not index.previous_valid[0]  # Earlier decision t=7 still lacks its endpoint.


def test_target_observation_presence_and_future_endpoint_are_independent():
    times = np.arange(10, dtype=np.int64) * 1000
    records = np.repeat("r", len(times))
    x = np.ones((len(times), 1, 25))
    y = np.zeros((len(times), 1, 3))
    scores = np.ones((len(times), 1))
    present = np.ones((len(times), 1), dtype=bool)
    present[8] = False
    working = np.zeros_like(present)
    working[8] = True
    index = index_windows(times, records, x, y, scores, present, target_working=working)
    assert np.array_equal(index.row_targets[index.sample_windows], [8])
    assert index.filters["incomplete_causal_history"] == 0

    present[8] = True
    x[7, 0, 0] = np.nan
    for future_value in (2., np.nan):
        x[8, 0, 0] = future_value
        repaired, ready = interpolate_short_gaps(x, times, records, 1, return_ready_at=True)
        index = index_windows(
            times, records, repaired, y, scores, present, ready_at=ready, target_working=working,
        )
        assert not len(index.sample_windows)
        assert index.filters["incomplete_causal_history"] == 1


@pytest.mark.parametrize("second_offset", [50, 200])
def test_training_round_trip(tmp_path, second_offset):
    rng = np.random.default_rng(13)
    local = np.tile(np.arange(100, dtype=np.int64), 2)
    t = (local + np.repeat([0, second_offset], 100)) * 1000
    features = rng.normal(size=(len(t), 1, 25))
    labels = rng.uniform(size=(len(t), 1, 3))
    # The held-out shift must not affect train-derived means.
    features[local >= 80] += 100
    enabled = np.ones((len(t), 1), dtype=bool)
    enabled[10] = False
    labels[11] = 10.
    labels[12] = np.nan
    multiplicity = np.ones((len(t), 1), dtype=int)
    multiplicity[8] = 0
    multiplicity[9] = 3
    data_file = tmp_path / "training.npz"
    np.savez(
        data_file, time_ms=t, record_id=np.repeat(["r1", "r2"], 100),
        source=np.repeat(["simulation", "measured"], 100),
        nozzle_ids=["one"], features=features, labels=labels, visibility=np.ones((len(t), 1)),
        target_enabled=enabled,
        feature_names=FEATURE_NAMES, feature_units=FEATURE_UNITS, action_units=ACTION_UNITS,
        label_source="fixture labels", visibility_source="fixture scores",
        sample_multiplicity=multiplicity, sampling_source="fixture selection",
    )
    with pytest.raises(ValueError, match="nozzle_ids"):
        load_training_npz(data_file, ["wrong"])
    records = load_training_npz(data_file, ["one"])
    assert isinstance(records, TrainingRecords)
    model_file = tmp_path / "model.pt"
    with pytest.raises(FileNotFoundError):
        load_predictor(model_file)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = train(
            records, [[0., 0., 0.]], [[1., 1., 1.]], model_file, warmup_start_lr=0.,
            rmse_monitor="standardised", max_gap_steps=1, max_epochs=1,
        )
        assert result["best_epoch"] == 1 and result["test"]["samples"] == 4
        assert result["split_scope"] == "source"
        training = result["subsets"]["training"]
        assert training["sample_count_by_source"] == {"measured": 72, "simulation": 70}
        assert training["ineligible_multiplicity"] == 19
        assert training["filters"]["disabled_target"] == 1
        assert training["filters"]["missing_enabled_label"] == 1
        assert training["filters"]["enabled_label_outside_working_domain"] == 1
        predictor = load_predictor(model_file)
        assert predictor.nozzle_ids == ("one",)
        assert predictor.max_gap_steps == 1
        assert np.allclose(predictor.feature_norm.mean, features[local < 79].mean(axis=(0, 1)))
        selected_rows = np.flatnonzero((local >= 8) & (local < 80))
        selected_rows = np.setdiff1d(selected_rows, [10, 11, 12])
        selected = np.repeat(labels[selected_rows, 0], multiplicity[selected_rows, 0], axis=0)
        assert np.allclose(predictor.label_norm.mean, selected.mean(axis=0))
        payload = torch.load(model_file, weights_only=True)
        assert payload["sampling_source"] == "fixture selection"
        assert payload["config"]["split_scope"] == "source"
        assert predictor.predict(features[:8].transpose(1, 0, 2)).shape == (1, 3)
    finally:
        torch.set_num_threads(previous_threads)


def test_old_checkpoint_requests_retraining(tmp_path):
    path = tmp_path / "old.pt"
    with path.open("wb") as handle:
        torch.save({"version": 1}, handle)
    with pytest.raises(ValueError, match="retrain"):
        load_predictor(path)
