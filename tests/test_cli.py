import json
from types import SimpleNamespace

import numpy as np

from mwir_control import cli
from mwir_control.cli import main
from mwir_control.features import ACTION_UNITS, FEATURE_NAMES, FEATURE_UNITS
from mwir_control.inputs import FRAME_UNITS


def _json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _session(path):
    stack_kinds = ["SCC-port", "SCC-starboard", "SCC-midships"]
    rows = []
    for region, kinds in (("SCC-F", stack_kinds + stack_kinds[:2]), ("SCC-A", stack_kinds)):
        rows.extend((region, kind, 0.0) for kind in kinds)
    rows.extend((f"SC_{i + 1}", "SC", 1.0 if i < 139 else -1.0) for i in range(278))
    nozzles = [
        dict(nozzle_id=f"test-{i}", region=region, kind=kind, position_m=[0, y, 0],
             mount_rotation=np.eye(3).tolist(), deflection_axis=[0, 1, 0],
             reference_pressure_mpa=0.3 if kind == "SC" else 0.827)
        for i, (region, kind, y) in enumerate(rows)
    ]
    _json(path / "registry.json", nozzles)
    (path / "nb.csv").write_text(
        "wavelength_um,response\n3.85,0\n3.875,0.5\n3.9,1\n3.925,0.5\n3.95,0\n",
        encoding="utf-8",
    )
    wavelengths = [3, 3.85, 3.875, 3.9, 3.925, 3.95, 5]
    _json(path / "session.json", {
        "schema_version": 1, "units": FRAME_UNITS, "channels": ["3-5", "3.9"],
        "nozzles_json": "registry.json", "frames": ["frame.json"],
        "spectral": {"wavelength_um": wavelengths, "narrow_band_um": [3.85, 3.95], "noise": [0.01, 0.005],
                     "broadband": {"source": "viewer_rsr_radiance_weight"},
                     "narrowband": {"source": "calibrated", "csv": "nb.csv"}},
        "water_nk": {"source": "viewer_25degC"}, "surface_emissivity_source": "viewer_grey",
    })
    stamp = 1_730_000_000_000
    _json(path / "frame.json", {
        "utc_ms": stamp, "record_id": "test", "source": "test-camera", "actual_source": "test-actual",
        "data_source": "simulation",
        "frame_npz": "frame.npz",
        "camera": {"shape": [3, 3], "fx": 10, "fy": 10, "cx": 1, "cy": 1,
                   "origin_m": [1000, 1000, 150], "camera_to_enu": [[1, 0, 0], [0, -1, 0], [0, 0, -1]]},
        "pose": {"ship_origin_m": [0, 0, 0], "ship_to_enu": np.eye(3).tolist()},
        "environment": [20, 15000, 2, 18], "vessel": [0, 90, 0, 0],
        "sun": {"elevation_deg": 45, "azimuth_deg": 90}, "roi": {"source": "masks"},
        "pixel_area_m2": 0.001,
        "boundary": {"atmosphere": {"source": "kruse", "path_m": 438}, "solar": {"source": "calibrated"},
                     "sea_reflectance": {"source": "calibrated_pixels"}, "sea_emissivity": 0.98, "plumes": []},
    })
    regions = ["Sea", *dict.fromkeys(region for region, _, _ in rows)]
    masks = np.ones((len(regions), 3, 3), dtype=bool)
    masks[0, :2] = False
    masks[1:, 2] = False
    ids = [nozzle["nozzle_id"] for nozzle in nozzles]
    off = np.zeros((len(ids), 3))
    enabled = np.zeros(len(ids), dtype=bool)
    frames = np.stack([np.full((3, 3), 2.0), np.full((3, 3), 0.1)])
    fields = dict(utc_ms=stamp, record_id="test", channels=["3-5", "3.9"], channel_utc_ms=[stamp, stamp + 5],
                  observed=frames, valid=np.ones(frames.shape, dtype=bool),
                  actual_utc_ms=stamp, actual_record_id="test", state_utc_ms=stamp, boundary_utc_ms=stamp,
                  nozzle_ids=ids, actual_actions=off, actual_enabled=enabled, roi_ids=regions, roi_masks=masks,
                  solar_irradiance=np.ones(len(wavelengths)), sea_reflectance=np.zeros((3, 3)))
    np.savez(path / "frame.npz", **fields)
    np.savez(path / "candidate.npz", nozzle_ids=ids, values=off, enabled=enabled)
    return fields


def test_cli_forward_score_and_action_time_check(tmp_path, capsys):
    fields = _session(tmp_path)
    session = str(tmp_path / "session.json")
    result_path = tmp_path / "forward.npz"
    assert main(["validate", session]) == 0
    assert json.loads(capsys.readouterr().out)["nozzles"] == 286
    assert main(["forward", session, "frame.json", str(result_path)]) == 0
    with np.load(result_path, allow_pickle=False) as result:
        assert result["atmosphere_source"].item().lower().startswith("kruse")
        assert result["modelled"].shape == (2, 3, 3)
        assert result["observed_roi"].shape == (2, 281, 5)
        np.testing.assert_allclose(result["observed_roi"][:, 0, 0], [2.0, 0.1])
        assert not result["fog_chord_m"].any()
        np.testing.assert_allclose(result["residual"], fields["observed"] - result["modelled"])
    candidate = str(tmp_path / "candidate.npz")
    report = tmp_path / "scores.json"
    assert main(["evaluate", session, "frame.json", str(report), candidate]) == 0
    result = json.loads(report.read_text(encoding="utf-8"))["candidates"][0]
    assert len(result["scores"]) == 280 and sum(result["flow_l_min"]) == 0
    np.savez(tmp_path / "frame.npz", **{**fields, "actual_utc_ms": fields["utc_ms"] + 1000})
    assert main(["validate", session]) == 2
    assert "actual_utc_ms" in capsys.readouterr().err


def test_cli_align_and_metrics(tmp_path, capsys):
    raw, policy, aligned = [tmp_path / name for name in ("raw.npz", "policy.json", "aligned.npz")]
    np.savez(raw, time_ms=[1000, 2000], wb_time_ms=[990, 1990], nb_time_ms=[1005, 2005])
    _json(policy, {"mode": "offline", "channel_tolerance_ms": 30})
    assert main(["align", str(policy), str(raw), str(aligned)]) == 0
    with np.load(aligned) as data:
        assert data["valid"].all() and (data["jitter_ms"] == 15).all()
    _json(policy, {"mode": "causal", "max_age_ms": {"wb": 30, "nb": 30}})
    assert main(["align", str(policy), str(raw), str(aligned)]) == 0
    with np.load(aligned) as data:
        assert not data["valid"].any()

    np.savez(tmp_path / "latency.npz", timestamps_ns=np.array([0, 1_000_000_000], dtype=np.int64),
             source="unit-test", latency_units="ms", io=[1, 2], phys=[3, 4],
             prep=[5, 6], lstm=[7, 8], vso=[0, 10], guard=[11, 12], tx=[13, 14])
    manifest, output = tmp_path / "eval.json", tmp_path / "metrics.json"
    settings = {"schema_version": 1, "latency": {"npz": "latency.npz"}}
    _json(manifest, settings)
    args = ["metrics", str(manifest), str(output)]
    assert main(args) == 0
    result = json.loads(output.read_text(encoding="utf-8"))["latency"]
    assert result["per_cycle_ms"] == [40, 56]
    assert result["mean_ms"] == 48
    _json(manifest, {**settings, "scaled": {}})
    previous = output.read_bytes()
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "scaled" in err and "no longer supported" in err
    assert output.read_bytes() == previous


def _enable_schedule(tmp_path, fields, enabled, *, available_offset_ms=0):
    session = tmp_path / "session.json"
    data = json.loads(session.read_text(encoding="utf-8"))
    _json(session, {**data, "enable_schedule_npz": "enable.npz"})
    stamp = fields["utc_ms"]
    np.savez(tmp_path / "enable.npz", target_time_ms=np.array([stamp + 1000], dtype=np.int64),
             available_time_ms=np.array([stamp + available_offset_ms], dtype=np.int64),
             record_id=["test"], source=["test-schedule"], nozzle_ids=fields["nozzle_ids"], enabled=enabled[None])
    return session


def test_cli_predict_uses_scheduled_enable_and_causal_history(tmp_path, monkeypatch, capsys):
    fields = _session(tmp_path)
    cutoff = fields["utc_ms"]
    history_path = tmp_path / "history.npz"
    output = tmp_path / "actions.npz"
    enabled = np.zeros(286, dtype=bool)
    enabled[-1] = True
    windows = []

    def predict(batch):
        windows.append(batch.copy())
        return np.array([[0.3, 0.7, 1.0]])

    monkeypatch.setattr(cli, "_predictor", lambda *args: SimpleNamespace(max_gap_steps=1, predict=predict))
    history = np.full((8, 286, 25), np.nan)
    history[:, -1, :] = np.arange(8)[:, None]
    history[6, -1, 0] = np.nan
    payload = dict(time_ms=cutoff - 1000 * np.arange(7, -1, -1, dtype=np.int64), decision_time_ms=np.int64(cutoff),
                   record_id="test", history=history, nozzle_ids=fields["nozzle_ids"],
                   feature_names=FEATURE_NAMES, feature_units=FEATURE_UNITS)
    np.savez(history_path, **payload)
    args = ["predict", str(tmp_path / "session.json"), "checkpoint.pt", str(history_path), str(output)]
    assert main(args) == 2
    assert "enable_schedule_npz" in capsys.readouterr().err

    _enable_schedule(tmp_path, fields, enabled, available_offset_ms=1)
    assert main(args) == 2
    assert "unavailable at the decision" in capsys.readouterr().err

    _enable_schedule(tmp_path, fields, enabled)
    assert main(args) == 0
    assert windows[0].shape == (1, 8, 25) and windows[0][0, 6, 0] == 6
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["enabled"], enabled)
        np.testing.assert_allclose(data["values"][-1], [0.3, 0.7, 1])
        assert np.all(data["values"][:-1] == 0)
        assert np.isnan(data["raw"][:-1]).all()
        assert data["target_time_ms"].item() == cutoff + 1000
        assert data["enable_available_time_ms"].item() == cutoff
        assert data["enable_source"].item() == "test-schedule"
    history[-1, -1, 0] = np.nan
    np.savez(history_path, **payload)
    assert main(args) == 2
    assert "Incomplete causal history" in capsys.readouterr().err
    assert len(windows) == 1


def test_cli_replay_logs_all_off_schedule_and_enable_provenance(tmp_path, monkeypatch):
    fields = _session(tmp_path)
    fields["roi_masks"][-1] = False
    np.savez(tmp_path / "frame.npz", **fields)
    enabled = np.zeros(286, dtype=bool)
    session = _enable_schedule(tmp_path, fields, enabled)
    output = tmp_path / "replay.jsonl"

    def should_not_infer(_):
        raise AssertionError("All-off replay must not infer continuous working settings")

    monkeypatch.setattr(cli, "_predictor", lambda *args: SimpleNamespace(max_gap_steps=1, predict=should_not_infer))
    assert main(["replay", str(session), "checkpoint.pt", str(output)]) == 0
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["proposal_status"] == "planned"
    assert row["requested_enable"]["source"] == "test-schedule"
    assert row["requested_enable"]["available_timestamp_ns"] == fields["utc_ms"] * 1_000_000
    assert row["plan"]["target_timestamp_ns"] == (fields["utc_ms"] + 1000) * 1_000_000
    assert not any(row["plan"]["final_enabled"]) and row["plan"]["raw"] is None
    assert row["plan"]["virtual_evaluations"] == 1 and row["plan"]["vso"] == []
    assert row["plan"]["final_scores"] == {}


def test_prepare_training_preserves_label_enable_and_data_source(tmp_path, capsys):
    fields = _session(tmp_path)
    label_path, output = tmp_path / "labels.npz", tmp_path / "training.npz"
    labels = dict(time_ms=np.array([fields["utc_ms"]], dtype=np.int64), record_id=["test"],
                  nozzle_ids=fields["nozzle_ids"], action_units=ACTION_UNITS, labels=np.zeros((1, 286, 3)),
                  visibility=np.zeros((1, 286)), label_source="external-labels", visibility_source="external-scores")
    args = ["prepare-training", str(tmp_path / "session.json"), str(label_path), str(output)]
    np.savez(label_path, **labels)
    assert main(args) == 2
    assert "target_enabled" in capsys.readouterr().err
    np.savez(label_path, **labels, target_enabled=np.zeros((1, 286), dtype=bool))
    assert main(args) == 0
    with np.load(output, allow_pickle=False) as data:
        assert data["source"].tolist() == ["simulation"]
        assert data["capture_source"].tolist() == ["test-camera"]
        assert not data["target_enabled"].any()
