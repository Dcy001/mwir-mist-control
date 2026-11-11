"""Command-line entry points for physical evaluation, learning and replay."""

import argparse
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .inputs import (CHANNELS, SessionInputs, action_from_npz, field, load_enable_schedule,
                     nonempty_string, read_npz, session_registry)
from .resources import PublicResources, read_json


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def inspect_resources(args):
    resources = PublicResources(args.root) if args.root else PublicResources()
    resources.verify()
    data = {
        "commit": resources.manifest["commit"],
        "ship_triangles": len(resources.mesh("ship_mesh").triangles),
        "sea_triangles": len(resources.mesh("sea_mesh").triangles),
        "broadband_points": len(resources.broadband_response()),
        "water_index_points": len(resources.water_nk()),
        "stack_outlets": list(resources.stack_outlets()),
    }
    print(json.dumps(data, indent=2))


def session_frames(session):
    """Validate per-record ordering. Gaps reset the controller's history."""
    paths = field(session.config, "frames", "session")
    if not isinstance(paths, list) or not paths:
        raise ValueError("session.frames must be a non-empty list of frame manifests")
    seen, previous = set(), None
    for path in paths:
        frame = session.load_frame(path)
        if previous and previous.record_id == frame.record_id:
            if (previous.utc_ms >= frame.utc_ms or previous.source != frame.source
                    or previous.data_source != frame.data_source):
                raise ValueError(f"{path}: observation source or data category changed, or timestamps are not increasing")
        elif frame.record_id in seen:
            raise ValueError(f"{path}: record {frame.record_id} appears in separate blocks")
        seen.add(frame.record_id)
        previous = frame
        yield frame


def validate_session(args):
    session = SessionInputs(args.session)
    schedule = load_enable_schedule(args.session, session.registry, session.config) if session.config.get("enable_schedule_npz") else {}
    count = sum(1 for _ in session_frames(session))
    print(json.dumps({"frames": count, "nozzles": len(session.registry),
                      "enable_schedule_targets": len(schedule),
                      "roi_ids": ["Sea", *session.registry.region_ids]}))


def _score(radiance, frame, regions=None):
    from .control import visibility

    if regions is not None and not regions:
        return {}
    masks = frame.masks if regions is None else {
        name: mask for name, mask in frame.masks.items() if name == "Sea" or name in regions
    }
    return visibility(radiance, masks, frame.valid, frame.noise, frame.glint)


def _forward(args):
    from .roi import summarise_rois

    session = SessionInputs(args.session)
    frame = session.load_frame(args.frame)
    start = perf_counter()
    result = frame.model.evaluate(frame.actual)
    elapsed_ms = (perf_counter() - start) * 1000
    support = frame.valid & np.isfinite(result.frames)
    residual = np.where(support, frame.frames - result.frames, np.nan)
    write_npz(args.output, time_ms=frame.utc_ms, record_id=frame.record_id, source=frame.source,
              actual_source=frame.actual_source, channels=CHANNELS, nozzle_ids=session.registry.ids,
              actual_actions=frame.actual.values, actual_enabled=frame.actual.enabled,
              modelled=result.frames, residual=residual,
              valid=support, fog_region=result.fog_region, fog_chord_m=result.fog_chord_m,
              region_ids=result.region_names, atmosphere_source=result.atmosphere_source,
              background_negative_samples=result.background_negative_samples, physical_ms=elapsed_ms,
              roi_ids=list(frame.masks), roi_statistics=["mean", "sample_std", "max", "hotspot_count", "hotspot_area_m2"],
              observed_roi=summarise_rois(frame.frames, frame.masks, frame.valid, frame.pixel_area_m2),
              modelled_roi=summarise_rois(result.frames, frame.masks, support, frame.pixel_area_m2),
              residual_roi=summarise_rois(residual, frame.masks, support, frame.pixel_area_m2),
              glint_risk=frame.glint)


def _evaluate(args):
    session = SessionInputs(args.session)
    frame = session.load_frame(args.frame)
    records = []
    for path in args.actions:
        action = action_from_npz(path, session.registry)
        session.registry.validate_action(action, feasible=True)
        start = perf_counter()
        result = frame.model.evaluate(action)
        scores = _score(result.frames, frame)
        records.append({
            "action_file": str(path), "action": action, "scores": scores,
            "flow_l_min": session.registry.flows(action),
            "atmosphere_source": result.atmosphere_source,
            "background_negative_samples": result.background_negative_samples,
            "evaluation_ms": (perf_counter() - start) * 1000,
        })
    write_json(args.output, {"record_id": frame.record_id, "boundary_time_ms": frame.utc_ms,
                             "nozzle_ids": session.registry.ids, "candidates": records})


def _prepare_training(args):
    from .data import keyed_indices
    from .features import ACTION_UNITS, FEATURE_NAMES, FEATURE_UNITS
    from .learning import load_training_npz

    external = read_npz(args.labels)
    target_enabled = field(external, "target_enabled", f"{args.labels} (add explicit label-time enable state)")
    session = SessionInputs(args.session)
    ids = session.registry.ids
    if tuple(external["nozzle_ids"]) != ids or tuple(external["action_units"]) != ACTION_UNITS:
        raise ValueError("External labels require registry nozzle_ids and documented action_units")
    if target_enabled.dtype != np.bool_ or target_enabled.shape != (len(external["time_ms"]), len(ids)):
        raise ValueError("External target_enabled must be Boolean [timestamps, registered nozzles]")
    times, records, sources, capture_sources, features = [], [], [], [], []
    for frame in session_frames(session):
        if frame.data_source is None:
            raise ValueError(f"Record {frame.record_id}: add frame.data_source = simulation or measured for training")
        result = frame.model.evaluate(frame.actual)
        times.append(frame.utc_ms)
        records.append(frame.record_id)
        sources.append(frame.data_source)
        capture_sources.append(frame.source)
        features.append(frame.features(result.frames, frame.actual, session.registry))
    times = np.asarray(times, dtype=np.int64)
    records = np.asarray(records)
    joined = keyed_indices(times, records, external["time_ms"], external["record_id"])
    payload = dict(time_ms=times, record_id=records, source=sources, capture_source=capture_sources, nozzle_ids=ids,
                   feature_names=FEATURE_NAMES, feature_units=FEATURE_UNITS, action_units=ACTION_UNITS,
                   features=np.stack(features), labels=external["labels"][joined],
                   target_enabled=target_enabled[joined],
                   visibility=external["visibility"][joined], label_source=external["label_source"],
                   visibility_source=external["visibility_source"])
    for optional in ("nozzle_present", "sample_multiplicity"):
        if optional in external:
            payload[optional] = external[optional][joined]
    if "sampling_source" in external:
        payload["sampling_source"] = external["sampling_source"]
    write_npz(args.output, **payload)
    load_training_npz(args.output, ids)


def _train(args):
    from .learning import load_training_npz, train

    registry = session_registry(args.session)
    records = load_training_npz(args.training, registry.ids)
    result = train(records, registry.lower, registry.upper, args.checkpoint,
                   warmup_start_lr=args.warmup_start_lr, rmse_monitor=args.rmse_monitor,
                   max_gap_steps=args.max_gap_steps, device=args.device, split_scope=args.split_scope)
    write_json(args.report or str(args.checkpoint) + ".json", result)


def _predictor(registry, checkpoint, device):
    from .learning import load_predictor
    predictor = load_predictor(checkpoint, device=device)
    if tuple(predictor.nozzle_ids) != registry.ids:
        raise ValueError("Checkpoint nozzle order does not match the session registry")
    return predictor


def _next_enable(schedule, current):
    target = current.successor()
    if target not in schedule:
        raise ValueError(f"Enable schedule is missing record {target.record_id} target {target.timestamp_ns // 1_000_000} ms")
    return schedule[target]


def _predict(args):
    from .cycle import CycleKey
    from .data import repair_causal_window, utc_milliseconds
    from .features import FEATURE_NAMES, FEATURE_UNITS, WINDOW_STEPS

    registry = session_registry(args.session)
    schedule = load_enable_schedule(args.session, registry)
    predictor = _predictor(registry, args.checkpoint, args.device)
    data = read_npz(args.history)
    history = field(data, "history", f"{args.history} (supply raw, timestamped features in place of windows)")
    if (tuple(data["nozzle_ids"]) != registry.ids or tuple(data["feature_names"]) != FEATURE_NAMES
            or tuple(data["feature_units"]) != FEATURE_UNITS):
        raise ValueError("History IDs, feature names and units must match the session/checkpoint contract")
    times = utc_milliseconds(field(data, "time_ms", args.history))
    cutoff = field(data, "decision_time_ms", args.history)
    if (not len(times) or cutoff.ndim or cutoff.dtype.kind != "i" or times[-1] != cutoff.item()
            or (times % 1000).any() or not (np.diff(times) == 1000).all()):
        raise ValueError("History must be contiguous on the 1 Hz UTC grid and end at decision_time_ms")
    if history.shape != (len(times), len(registry), len(FEATURE_NAMES)) or history.dtype.kind != "f" or np.isinf(history).any():
        raise ValueError("history must be floating [time, nozzles, 25], using NaN for missing features")
    if "feature_valid" in data:
        valid = data["feature_valid"]
        if valid.shape != history.shape or valid.dtype != np.bool_:
            raise ValueError("feature_valid must be Boolean and have the same shape as history")
        history = np.where(valid, history, np.nan)
    record = field(data, "record_id", args.history)
    if record.ndim or record.dtype.kind != "U":
        raise ValueError("History record_id must be a Unicode scalar identifying one continuous record")
    key = CycleKey(nonempty_string(record.item(), "record_id"), int(cutoff.item()) * 1_000_000)
    plan = _next_enable(schedule, key)
    raw = np.full((len(registry), 3), np.nan)
    if plan.enabled.any():
        repaired = repair_causal_window(history, predictor.max_gap_steps)
        if len(repaired) != WINDOW_STEPS:
            raise ValueError(f"Prediction needs {WINDOW_STEPS} consecutive steps for enabled nozzles")
        ready = np.isfinite(repaired).all(axis=(0, 2))
        if not ready[plan.enabled].all():
            unavailable = [name for name, missing in zip(registry.ids, plan.enabled & ~ready) if missing]
            raise ValueError(f"Incomplete causal history for enabled nozzles: {', '.join(unavailable)}")
        raw[plan.enabled] = predictor.predict(np.swapaxes(repaired[:, plan.enabled], 0, 1))
    projected = registry.project(raw, plan.enabled)
    write_npz(args.output, nozzle_ids=registry.ids, values=projected.values,
              enabled=projected.enabled, raw=raw, record_id=key.record_id,
              decision_time_ms=times[-1], target_time_ms=plan.key.timestamp_ns // 1_000_000,
              enable_available_time_ms=plan.available_timestamp_ns // 1_000_000, enable_source=plan.source)


def _replay(args):
    from .cycle import ActualReceipt, CycleController, CycleKey, Observation, cycle_log

    session = SessionInputs(args.session)
    schedule = load_enable_schedule(args.session, session.registry, session.config)
    predictor = _predictor(session.registry, args.checkpoint, args.device)

    def score_enabled(radiance, frame, action):
        regions = {region for region, enabled in zip(session.registry.regions, action.enabled) if enabled}
        return _score(radiance, frame, regions)

    controller = CycleController(session.registry, lambda action, frame: frame.model.evaluate(action).frames,
                                 lambda obs, model, action, frame: frame.features(model, action, session.registry),
                                 predictor.predict, score_enabled, max_gap_steps=predictor.max_gap_steps)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = iter(session_frames(session))
    with output.open("w", encoding="utf-8") as stream:
        while True:
            start = perf_counter()
            frame = next(frames, None)
            if frame is None:
                break
            boundary_io_ms = (perf_counter() - start) * 1000
            key = CycleKey(frame.record_id, frame.utc_ms * 1_000_000)
            enable_plan = _next_enable(schedule, key)
            feedback = controller.ingest(Observation(key, frame.frames, frame.valid, frame, frame.source),
                                         ActualReceipt(key, session.registry.ids, frame.actual, frame.actual_source))
            plan = controller.propose(enable_plan)
            elapsed = (perf_counter() - start) * 1000
            entry = cycle_log(feedback, plan, requested_enable=enable_plan, proposal_status=controller.proposal_status)
            entry["boundary_io_ms"] = boundary_io_ms
            entry["replay_compute_ms"] = elapsed
            entry["actual_flow_l_min"] = session.registry.flows(frame.actual)
            entry["checkpoint"] = str(args.checkpoint)
            if args.feedback_dir:
                path = Path(args.feedback_dir) / f"frame-{stream.tell():016d}.npz"
                write_npz(path, time_ms=frame.utc_ms, record_id=frame.record_id, nozzle_ids=session.registry.ids,
                          channels=CHANNELS, modelled=feedback.modelled, residual=feedback.residual,
                          features=feedback.features)
                entry["feedback_file"] = str(path)
            stream.write(json.dumps(json_value(entry), allow_nan=False) + "\n")
            stream.flush()


def _metrics(args):
    from .evaluation_io import evaluate_manifest
    write_json(args.output, evaluate_manifest(args.manifest))


def _align(args):
    from .data import align_channels_offline, causal_indices, correct_clock

    data = read_npz(args.input)
    policy = read_json(args.policy)
    stamp = {}
    for name in ("wb", "nb", *policy.get("streams", [])):
        raw = data[f"{name}_time_ms"]
        anchor = f"{name}_anchor_utc_ms"
        stamp[name] = correct_clock(raw, data[f"{name}_anchor_raw_ms"], data[anchor]) if anchor in data else raw
    grid = data["time_ms"]
    if policy["mode"] == "offline":
        alignment = align_channels_offline(grid, stamp["wb"], stamp["nb"], policy["channel_tolerance_ms"])
        payload = dict(wb_index=alignment.wb, nb_index=alignment.nb,
                       jitter_ms=alignment.jitter_ms, valid=alignment.valid)
    elif policy["mode"] == "causal":
        payload = {name + "_index": causal_indices(grid, stamp[name], policy["max_age_ms"][name])
                   for name in ("wb", "nb")}
        wb, nb = payload["wb_index"], payload["nb_index"]
        matched = (wb >= 0) & (nb >= 0)
        pair_valid = matched.copy()
        jitter = np.full(len(grid), np.nan)
        jitter[matched] = np.abs(stamp["wb"][wb[matched]] - stamp["nb"][nb[matched]])
        pair_valid[matched] = jitter[matched] < 20
        payload["valid"] = pair_valid
        payload["jitter_ms"] = jitter
    else:
        raise ValueError("Alignment policy mode must be offline or causal")
    for name in policy.get("streams", []):
        payload[name + "_index"] = causal_indices(grid, stamp[name], policy["max_age_ms"][name])
    write_npz(args.output, time_ms=grid, **payload)


def parser():
    cli = argparse.ArgumentParser(prog="mwir-control", description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)

    def command(name, callback, help_text, *positional):
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(callback=callback)
        for key in positional:
            sub.add_argument(key)
        return sub

    command("resources", inspect_resources, "verify pinned public assets").add_argument("--root")
    command("align", _align, "synchronise raw stream timestamps and export index maps", "policy", "input", "output")
    command("validate", validate_session, "validate session records and physical inputs", "session")
    command("forward", _forward, "evaluate recorded actual settings and calculate residuals", "session", "frame", "output")
    command("evaluate", _evaluate, "score explicitly provided candidates", "session", "frame", "output").add_argument("actions", nargs="+")
    command("prepare-training", _prepare_training, "derive residual features and join external labels", "session", "labels", "output")
    fit = command("train", _train, "train and save the best shared LSTM", "session", "training", "checkpoint")
    fit.add_argument("--warmup-start-lr", required=True, type=float)
    fit.add_argument("--rmse-monitor", required=True, choices=("standardised", "physical"))
    fit.add_argument("--max-gap-steps", required=True, type=int)
    fit.add_argument("--split-scope", choices=("source", "global"), default="source",
                     help="chronological split per data source (default) or across the pooled UTC timeline")
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--report")
    command("predict", _predict, "infer and project next-cycle actions from timestamped raw history", "session", "checkpoint", "history", "output").add_argument("--device", default="cpu")
    replay = command("replay", _replay, "replay actual feedback and virtual next-cycle selection", "session", "checkpoint", "output")
    replay.add_argument("--device", default="cpu")
    replay.add_argument("--feedback-dir")
    command("metrics", _metrics, "calculate metrics from a trial or evaluation manifest", "manifest", "output")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        args.callback(args)
    except (ValueError, KeyError, TypeError, OSError, ModuleNotFoundError) as exc:
        print(f"mwir-control {args.command}: {exc}", file=sys.stderr)
        return 2
    return 0
