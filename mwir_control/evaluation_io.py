"""Load and verify Appendix H evaluation records for the CLI."""

from pathlib import Path

import numpy as np

from .inputs import CHANNELS, absolute, nonempty_string, read_npz, session_registry
from .metrics import LATENCY_STAGES, Trial, compare_trials, latency_metrics, prediction_metrics, water_metrics
from .resources import read_json


def _string(data, name):
    value = data[name]
    if value.ndim or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode scalar")
    return nonempty_string(value.item(), name)


def _times(data, *, regular=False):
    times = data["timestamps_ns"]
    if (times.ndim != 1 or not times.size or times.dtype != np.int64
            or np.any(np.diff(times) <= 0)):
        raise ValueError("timestamps_ns must be a nonempty increasing int64 vector")
    if regular and not np.all(np.diff(times) == 1_000_000_000):
        raise ValueError("This evaluation requires complete 1 Hz timestamps")
    return times


def _trial(data):
    if tuple(data["channels"]) != CHANNELS or _string(data, "radiance_units") != "W m-2 sr-1":
        raise ValueError("Trial channels and radiance units must be ['3-5','3.9'] and W m-2 sr-1")
    _string(data, "source")
    temperature = data.get("temperature")
    if temperature is not None:
        unit = _string(data, "temperature_units")
        if unit not in {"K", "degC"}:
            raise ValueError("Trial temperatures require K or degC")
        if unit == "K":
            temperature = temperature - 273.15
    return Trial(_times(data), _string(data, "grid_id"), data["radiance"], data["valid"], temperature)


def _match_nozzles(data, registry):
    if tuple(data["nozzle_ids"]) != registry.ids:
        raise ValueError("Evaluation nozzle_ids do not match the registry order")


def evaluate_manifest(path):
    path = Path(path).resolve()
    config = read_json(path)
    if "scaled" in config:
        raise ValueError("The legacy 'scaled' evaluation section is no longer supported. Remove it from the manifest.")
    if config.get("schema_version") != 1:
        raise ValueError("Evaluation schema_version must be 1")
    results = {}

    def arrays(relative):
        return read_npz(absolute(path.parent, relative))

    if "prediction" in config:
        spec = config["prediction"]
        data = arrays(spec["npz"])
        registry = session_registry(absolute(path.parent, spec["session"]))
        _match_nozzles(data, registry)
        t = _times(data)
        if len(t) != len(data["predicted"]) or len(t) != len(data["labels"]):
            raise ValueError("Prediction/label arrays must share timestamps_ns")
        if tuple(data["action_units"]) != ("MPa", "1", "deg"):
            raise ValueError("Prediction action_units must be ['MPa','1','deg']")
        if "target_enabled" not in data:
            raise ValueError("Prediction evaluation NPZ requires target_enabled [T,N] Boolean action states")
        results["prediction"] = {
            "nozzle_ids": registry.ids,
            "prediction_source": _string(data, "prediction_source"),
            "label_source": _string(data, "label_source"),
            **prediction_metrics(data["predicted"], data["labels"], registry.lower, registry.upper,
                                 target_enabled=data["target_enabled"]),
        }
    if "comparison" in config:
        spec = config["comparison"]
        baseline = _trial(arrays(spec["no_mist"]))
        strategies = {name: _trial(arrays(value)) for name, value in spec["strategies"].items()}
        support = arrays(spec["support_npz"])
        if _string(support, "grid_id") != baseline.grid_id or not np.array_equal(_times(support), baseline.timestamps_ns):
            raise ValueError("ROI and glint support must share the trial grid and timestamps")
        masks = support["roi_masks"]
        ids = support["roi_ids"]
        if (ids.ndim != 1 or ids.dtype.kind != "U" or len(set(ids.tolist())) != len(ids)
                or masks.ndim not in (3, 4) or len(masks) != len(ids)):
            raise ValueError("Evaluation ROI masks must be [R,H,W] or [R,T,H,W] keyed by unique roi_ids")
        results["comparison"] = compare_trials(
            baseline, strategies, dict(zip(ids, masks)), support["pixel_area_m2"],
            support["noise"], support["glint"], spec["day_start_ns"], spec["epsilon"],
            roi_scope=spec.get("roi_scope"),
        )
    if "water" in config:
        spec = config["water"]
        registry = session_registry(absolute(path.parent, spec["session"]))
        data = arrays(spec["npz"])
        _match_nozzles(data, registry)
        _string(data, "source")
        _string(data, "full_load_source")
        if _string(data, "flow_units") != "L/min" or len(_times(data, regular=True)) != len(data["flow_l_min"]):
            raise ValueError("Water data require L/min and one rate per complete 1 Hz timestamp")
        results["water"] = water_metrics(data["flow_l_min"], data["full_load_l_min"])
    if "latency" in config:
        data = arrays(config["latency"]["npz"])
        _string(data, "source")
        times = _times(data)
        stages = {name: data[name] for name in LATENCY_STAGES}
        if _string(data, "latency_units") != "ms" or any(values.shape != times.shape for values in stages.values()):
            raise ValueError("Latency data require ms and one duration per stage per timestamp")
        results["latency"] = latency_metrics(stages)
    if not results:
        raise ValueError("Evaluation manifest must contain prediction, comparison, water or latency")
    return results
