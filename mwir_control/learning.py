"""Appendices E and F training, inference and checkpoint handling."""
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import contiguous_rows, interpolate_short_gaps, utc_milliseconds
from .features import ACTION_NAMES, ACTION_UNITS, FEATURE_NAMES, FEATURE_UNITS, WINDOW_STEPS


HIDDEN_SIZES = (128, 64, 32)
DROPOUT = 0.25
J_LIMIT = 3.0
CHECKPOINT_VERSION = 2


def _strings(values, shape, name):
    values = np.asarray(values)
    if values.shape != shape or values.dtype.kind != "U" or np.any(values == ""):
        raise ValueError(f"{name} must be non-empty Unicode strings of shape {shape}")
    return values


@dataclass(frozen=True)
class TrainingRecords:
    time_ms: np.ndarray
    record_id: np.ndarray
    source: np.ndarray
    nozzle_ids: tuple
    features: np.ndarray
    labels: np.ndarray
    target_enabled: np.ndarray
    visibility: np.ndarray
    nozzle_present: np.ndarray
    label_source: str
    visibility_source: str
    sample_multiplicity: np.ndarray = None
    sampling_source: str = None

    def __post_init__(self):
        times = utc_milliseconds(self.time_ms)
        if not len(times):
            raise ValueError("training records are empty")
        t, n = len(times), len(self.nozzle_ids)
        ids = _strings(self.nozzle_ids, (n,), "nozzle_ids")
        if not n or len(set(ids)) != n:
            raise ValueError("nozzle_ids must be nonempty and unique")
        records = _strings(self.record_id, (t,), "record_id")
        sources = _strings(self.source, (t,), "source")
        if not np.isin(sources, ("simulation", "measured")).all():
            raise ValueError("source must identify each row as simulation or measured")
        # Iteration validates record ordering, even when the records interleave.
        for _ in contiguous_rows(times, records):
            pass
        first_source = {}
        for record, time, source in zip(records, times, sources):
            phase_and_source = (int(time % 1000), source)
            if first_source.setdefault(record, phase_and_source) != phase_and_source:
                raise ValueError(f"record {record} must share one time-grid phase and one source")
        for name, shape in (
            ("features", (t, n, len(FEATURE_NAMES))),
            ("labels", (t, n, 3)),
            ("visibility", (t, n)),
        ):
            array = np.asarray(getattr(self, name))
            if array.shape != shape or array.dtype.kind != "f" or np.isinf(array).any():
                raise ValueError(f"{name} must be a floating array of shape {shape}, with NaN for missing values")
        present = np.asarray(self.nozzle_present)
        if present.shape != (t, n) or present.dtype != np.bool_:
            raise ValueError("nozzle_present must be Boolean [T,N]")
        target_enabled = np.asarray(self.target_enabled)
        if target_enabled.shape != (t, n) or target_enabled.dtype != np.bool_:
            raise ValueError("target_enabled must be Boolean [T,N]")
        if np.any(self.visibility < 0):
            raise ValueError("visibility scores must be non-negative")
        _strings(self.label_source, (), "label_source")
        _strings(self.visibility_source, (), "visibility_source")
        if (self.sample_multiplicity is None) != (self.sampling_source is None):
            raise ValueError("sample_multiplicity and sampling_source must be supplied together")
        if self.sample_multiplicity is not None:
            counts = np.asarray(self.sample_multiplicity)
            if counts.shape != (t, n) or counts.dtype.kind not in "iu" or np.any(counts < 0):
                raise ValueError("sample_multiplicity must be non-negative integers [T,N]")
            _strings(self.sampling_source, (), "sampling_source")


def load_training_npz(path, nozzle_ids):
    """Load aligned records with exact schema, units and registry order."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "time_ms", "record_id", "source", "nozzle_ids", "feature_names",
            "feature_units", "action_units", "features", "labels", "target_enabled", "visibility",
            "label_source", "visibility_source",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"training NPZ missing fields: {sorted(missing)}")
        for name, expected in (
            ("nozzle_ids", tuple(nozzle_ids)),
            ("feature_names", FEATURE_NAMES),
            ("feature_units", FEATURE_UNITS),
            ("action_units", ACTION_UNITS),
        ):
            actual = _strings(archive[name], (len(expected),), name)
            if tuple(actual) != expected:
                raise ValueError(f"{name} differs from required order or values")
        features = archive["features"]
        if "feature_valid" in archive:
            quality = archive["feature_valid"]
            if quality.shape != features.shape or quality.dtype != np.bool_:
                raise ValueError("feature_valid must be Boolean with the same shape as features")
            features = np.where(quality, features, np.nan)
        labels = archive["labels"]
        present = (
            archive["nozzle_present"] if "nozzle_present" in archive
            else np.ones(labels.shape[:2], dtype=bool)
        )
        counts = archive["sample_multiplicity"] if "sample_multiplicity" in archive else None
        sampling_source = (
            str(_strings(archive["sampling_source"], (), "sampling_source").item())
            if "sampling_source" in archive else None
        )
        return TrainingRecords(
            archive["time_ms"], archive["record_id"], archive["source"], tuple(nozzle_ids),
            features, labels, archive["target_enabled"], archive["visibility"], present,
            str(_strings(archive["label_source"], (), "label_source").item()),
            str(_strings(archive["visibility_source"], (), "visibility_source").item()),
            counts, sampling_source,
        )


@dataclass(frozen=True)
class Normalisation:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        mean, scale = np.asarray(self.mean), np.asarray(self.scale)
        if (mean.ndim != 1 or mean.shape != scale.shape or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("normalisation requires finite 1-D means and positive scales")

    @classmethod
    def fit(cls, values, eligible=None):
        mean = np.empty(values.shape[-1], dtype=np.float64)
        scale = np.empty_like(mean)
        for column in range(len(mean)):
            values_column = values[..., column]
            support = np.isfinite(values_column)
            if eligible is not None:
                support &= eligible
            finite = values_column[support]
            if not len(finite):
                raise ValueError(f"no finite training values for normalisation column {column}")
            mean[column] = finite.mean(dtype=np.float64)
            scale[column] = finite.std(dtype=np.float64) or 1.0
        return cls(mean, scale)

    def transform(self, values):
        return (values - self.mean) / self.scale

    def inverse(self, values):
        return values * self.scale + self.mean


def _split_times(times):
    distinct = np.unique(times)
    edge80 = int(0.80 * len(distinct))
    edge90 = int(0.90 * len(distinct))
    if not 0 < edge80 < edge90 < len(distinct):
        raise ValueError("not enough distinct timestamps for an 80/10/10 split")
    return (
        np.flatnonzero(times < distinct[edge80]),
        np.flatnonzero((times >= distinct[edge80]) & (times < distinct[edge90])),
        np.flatnonzero(times >= distinct[edge90]),
    )


def chronological_split(time_ms, source=None):
    """Split unique times 80/10/10 globally or within each explicit source."""
    times = utc_milliseconds(time_ms)
    if source is None:
        return _split_times(times)
    source = _strings(source, times.shape, "source")
    subsets = [[], [], []]
    for name in np.unique(source):
        rows = np.flatnonzero(source == name)
        try:
            source_subsets = _split_times(times[rows])
        except ValueError as exc:
            raise ValueError(f"source {name}: {exc}") from exc
        for combined, local in zip(subsets, source_subsets):
            combined.append(rows[local])
    if not len(source):
        raise ValueError("no rows for a source-stratified chronological split")
    return tuple(np.sort(np.concatenate(parts)) for parts in subsets)


@dataclass(frozen=True)
class WindowIndex:
    row_windows: np.ndarray
    row_targets: np.ndarray
    sample_windows: np.ndarray
    sample_nozzles: np.ndarray
    previous_valid: np.ndarray
    filters: dict


def index_windows(
    time_ms, record_id, features, labels, visibility, present, *,
    ready_at=None, target_working=None,
):
    """Index causal next-step samples and real same-nozzle predecessor windows."""
    good_step = present & np.isfinite(features).all(axis=-1)
    good_labels = np.isfinite(labels).all(axis=-1)
    if target_working is not None:
        good_labels &= target_working
    good_weights = (np.isfinite(visibility) | ~present).all(axis=-1) & present.any(axis=-1)
    windows, targets, sample_windows, sample_nozzles, previous = [], [], [], [], []
    offset = 0
    filters = {
        "no_continuous_history": int(good_labels.sum()),
        "incomplete_causal_history": 0,
        "missing_visibility": 0,
    }
    for rows in contiguous_rows(time_ms, record_id):
        if len(rows) <= WINDOW_STEPS:
            continue
        window_rows = np.lib.stride_tricks.sliding_window_view(rows, WINDOW_STEPS + 1)
        complete = np.lib.stride_tricks.sliding_window_view(
            good_step[rows[:-1]], WINDOW_STEPS, axis=0
        ).all(axis=-1)
        good_windows = complete
        if ready_at is not None:
            last_ready = np.lib.stride_tricks.sliding_window_view(
                ready_at[rows[:-1]], WINDOW_STEPS, axis=0,
            ).max(axis=-1)
            good_windows = complete & (last_ready <= window_rows[:, -2, None])
        target_ok = good_labels[window_rows[:, -1]]
        score_ok = good_weights[window_rows[:, -2], None]
        filters["no_continuous_history"] -= int(target_ok.sum())
        filters["incomplete_causal_history"] += int((target_ok & ~good_windows).sum())
        filters["missing_visibility"] += int((target_ok & good_windows & ~score_ok).sum())
        valid = target_ok & good_windows & score_ok
        when, which = np.nonzero(valid)
        earlier = np.maximum(when - 1, 0)
        previous.append((when > 0) & good_windows[earlier, which] & target_ok[earlier, which])
        windows.append(window_rows[:, :-1])
        targets.append(window_rows[:, -1])
        sample_windows.append(when + offset)
        sample_nozzles.append(which)
        offset += len(window_rows)
    if not windows:
        return WindowIndex(
            np.empty((0, WINDOW_STEPS), dtype=int), np.empty(0, dtype=int),
            np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=bool),
            filters,
        )
    return WindowIndex(
        np.concatenate(windows), np.concatenate(targets),
        np.concatenate(sample_windows), np.concatenate(sample_nozzles), np.concatenate(previous),
        filters,
    )


@dataclass(frozen=True)
class WindowData:
    features: np.ndarray
    labels: np.ndarray
    visibility: np.ndarray
    index: WindowIndex
    target_counts: dict
    discarded_multiplicity: int = 0

    def selected_labels(self):
        return self.labels[self.index.row_targets[self.index.sample_windows], self.index.sample_nozzles]

    def selected_rows(self):
        return self.index.row_targets[self.index.sample_windows]

    def feature_support(self):
        """Unique training rows/nozzles exposed in causal loss inputs."""
        index = self.index
        used = np.zeros(self.features.shape[:2], dtype=bool)
        predecessors = index.previous_valid
        for offset in range(WINDOW_STEPS):
            used[index.row_windows[index.sample_windows, offset], index.sample_nozzles] = True
            used[
                index.row_windows[index.sample_windows[predecessors] - 1, offset],
                index.sample_nozzles[predecessors],
            ] = True
        return used


def _prepare_subset(records, rows, max_gap_steps, lower, upper):
    times, record_ids = records.time_ms[rows], records.record_id[rows]
    present = records.nozzle_present[rows]
    features, ready_at = interpolate_short_gaps(
        records.features[rows], times, record_ids, max_gap_steps, eligible=present[..., None],
        return_ready_at=True,
    )
    labels, visibility = records.labels[rows], records.visibility[rows]
    enabled = records.target_enabled[rows]
    finite = np.isfinite(labels).all(axis=-1)
    in_domain = ((labels >= lower) & (labels <= upper)).all(axis=-1)
    working = enabled & finite & in_domain
    target_counts = {
        "enabled_targets": int(enabled.sum()),
        "disabled_target": int((~enabled).sum()),
        "missing_enabled_label": int((enabled & ~finite).sum()),
        "enabled_label_outside_working_domain": int((enabled & finite & ~in_domain).sum()),
    }
    index = index_windows(
        times, record_ids, features, labels, visibility, present,
        ready_at=ready_at, target_working=working,
    )
    if records.sample_multiplicity is None:
        return WindowData(features, labels, visibility, index, target_counts)
    multiplicity = records.sample_multiplicity[rows]
    targets = index.row_targets[index.sample_windows]
    counts = multiplicity[targets, index.sample_nozzles]
    requested = sum(map(int, multiplicity.ravel()))
    if requested > np.iinfo(np.intp).max:
        raise ValueError("sample multiplicity total exceeds addressable index range")
    repeats = np.repeat(np.arange(len(counts)), counts.astype(np.intp))
    discarded = requested - len(repeats)
    selected = WindowIndex(
        index.row_windows, index.row_targets, index.sample_windows[repeats],
        index.sample_nozzles[repeats], index.previous_valid[repeats],
        index.filters,
    )
    return WindowData(features, labels, visibility, selected, target_counts, discarded)


class _Samples(Dataset):
    def __init__(self, data, feature_norm, label_norm):
        self.data = data
        self.feature_norm = feature_norm
        self.label_norm = label_norm
        maximum = np.max(np.where(np.isfinite(data.visibility), data.visibility, -np.inf), axis=1)
        self.weights = (1 + 0.5 * np.clip(maximum / J_LIMIT, 0, 1)).astype(np.float32)

    def __len__(self):
        return len(self.data.index.sample_windows)

    def _features(self, when, nozzle):
        rows = self.data.index.row_windows[when]
        return self.feature_norm.transform(self.data.features[rows, nozzle]).astype(np.float32)

    def __getitem__(self, sample):
        index = self.data.index
        when, nozzle = index.sample_windows[sample], index.sample_nozzles[sample]
        x = self._features(when, nozzle)
        previous = index.previous_valid[sample]
        x_previous = self._features(when - 1, nozzle) if previous else np.zeros_like(x)
        y = self.data.labels[index.row_targets[when], nozzle]
        weight = self.weights[index.row_windows[when, -1]]
        return x, x_previous, bool(previous), self.label_norm.transform(y).astype(np.float32), int(nozzle), weight


class SprayLSTM(nn.Module):
    """Shared three-layer Appendix E.2 LSTM with a linear final-state readout."""

    def __init__(self):
        super().__init__()
        widths = (len(FEATURE_NAMES), *HIDDEN_SIZES)
        self.layers = nn.ModuleList([
            nn.LSTM(a, b, batch_first=True) for a, b in zip(widths[:-1], widths[1:])
        ])
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Linear(HIDDEN_SIZES[-1], len(ACTION_NAMES))

    def forward(self, windows):
        if windows.ndim != 3 or windows.shape[1:] != (WINDOW_STEPS, len(FEATURE_NAMES)):
            raise ValueError("LSTM input must have shape [batch,8,25]")
        sequence = windows
        for layer in self.layers:
            sequence, _ = layer(sequence)
            sequence = self.dropout(sequence)
        return self.output(sequence[:, -1])


def training_loss(predicted_z, targets_z, previous_z, have_previous, lower, upper, mean, scale, weight):
    """Table F.5 loss. Constraint and temporal terms are in physical units."""
    predicted = predicted_z * scale + mean
    projected = torch.minimum(torch.maximum(predicted, lower), upper)
    supervised = (predicted_z - targets_z).square().mean(dim=-1)
    constraint = (predicted - projected).square().sum(dim=-1)
    temporal = ((predicted_z - previous_z) * scale).square().sum(dim=-1)
    temporal = torch.where(have_previous, temporal, torch.zeros_like(temporal))
    return (weight * (supervised + 0.10 * constraint + 0.01 * temporal)).mean()


def learning_rate(epoch, max_epochs, warmup_start_lr):
    if epoch <= 5:
        return warmup_start_lr + (1e-3 - warmup_start_lr) * epoch / 5
    progress = (epoch - 5) / (max_epochs - 5)
    return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * progress))


def _validation_metrics(model, loader, mean, scale, device):
    z_sum = np.zeros(3, dtype=float)
    physical_sum = np.zeros(3, dtype=float)
    physical_absolute = np.zeros(3, dtype=float)
    count = 0
    model.eval()
    with torch.no_grad():
        for x, _, _, y, _, _ in loader:
            dz = model(x.to(device)) - y.to(device)
            du = dz * scale
            z_sum += dz.square().sum(dim=0).cpu().numpy()
            physical_sum += du.square().sum(dim=0).cpu().numpy()
            physical_absolute += du.abs().sum(dim=0).cpu().numpy()
            count += len(x)
    if not np.isfinite(physical_sum).all():
        raise FloatingPointError("non-finite validation prediction error")
    return {
        "samples": count, "standardised_rmse": float(np.sqrt(z_sum.mean() / count)),
        "physical_rmse": float(np.sqrt(physical_sum.mean() / count)),
        "physical_rmse_by_action": np.sqrt(physical_sum / count).tolist(),
        "physical_mae_by_action": (physical_absolute / count).tolist(),
    }


def _save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train(
    records, lower, upper, checkpoint_path, *, warmup_start_lr, rmse_monitor,
    max_gap_steps, device="cpu", max_epochs=180, seed=0, split_scope="source",
):
    """Run Table F.5 training, write the best checkpoint and evaluate held-out data."""
    if not np.isfinite(warmup_start_lr) or not 0 <= warmup_start_lr <= 1e-3:
        raise ValueError("warmup_start_lr must be in [0,0.001]")
    if rmse_monitor not in {"standardised", "physical"}:
        raise ValueError("rmse_monitor must be standardised or physical")
    if not isinstance(max_epochs, int) or not 1 <= max_epochs <= 180:
        raise ValueError("max_epochs must be in [1,180]")
    if split_scope not in {"global", "source"}:
        raise ValueError("split_scope must be global or source")
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (len(records.nozzle_ids), 3) or upper.shape != lower.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower >= upper)):
        raise ValueError("working bounds must be finite [N,3] with lower < upper")
    split = chronological_split(records.time_ms, records.source if split_scope == "source" else None)
    subsets = tuple(_prepare_subset(records, rows, max_gap_steps, lower, upper) for rows in split)
    split_report = {}
    for name, rows, subset in zip(("training", "validation", "test"), split, subsets):
        diagnostics = {**subset.target_counts, **subset.index.filters}
        if not len(subset.index.sample_windows):
            raise ValueError(
                f"no eligible eight-step windows in {name}: {diagnostics}, "
                f"ineligible_multiplicity={subset.discarded_multiplicity}"
            )
        sources = records.source[rows[subset.selected_rows()]]
        split_report[name] = {
            "sample_count_by_source": {
                str(source): int((sources == source).sum()) for source in np.unique(records.source[rows])
            },
            "ineligible_multiplicity": subset.discarded_multiplicity,
            "filters": diagnostics,
        }
    feature_norm = Normalisation.fit(subsets[0].features, subsets[0].feature_support())
    label_norm = Normalisation.fit(subsets[0].selected_labels())

    torch.manual_seed(seed)
    device = torch.device(device)
    model = SprayLSTM().to(device)
    generator = torch.Generator().manual_seed(seed)
    loaders = tuple(
        DataLoader(
            _Samples(subset, feature_norm, label_norm), batch_size=256,
            shuffle=index == 0, generator=generator,
        )
        for index, subset in enumerate(subsets)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
    )
    lower_t, upper_t, mean_t, scale_t = [
        torch.tensor(array, dtype=torch.float32, device=device)
        for array in (lower, upper, label_norm.mean, label_norm.scale)
    ]
    best_monitor = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, max_epochs + 1):
        lr = learning_rate(epoch, max_epochs, warmup_start_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        sum_loss, samples = 0.0, 0
        for batch in loaders[0]:
            x, previous_x, has_previous, y, nozzle, weight = (item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            all_predictions = model(torch.cat((x, previous_x[has_previous]), dim=0))
            prediction = all_predictions[:len(x)]
            previous_prediction = torch.zeros_like(prediction)
            previous_prediction[has_previous] = all_predictions[len(x):]
            loss = training_loss(
                prediction, y, previous_prediction, has_previous,
                lower_t[nozzle], upper_t[nozzle], mean_t, scale_t, weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            sum_loss += float(loss.detach()) * len(x)
            samples += len(x)
        validation = _validation_metrics(model, loaders[1], mean_t, scale_t, device)
        history.append({
            "epoch": epoch, "learning_rate": lr, "training_loss": sum_loss / samples,
            "validation": validation,
        })
        monitor = validation[f"{rmse_monitor}_rmse"]
        if monitor < best_monitor - 1e-4:
            best_monitor, best_epoch = monitor, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if epoch - best_epoch >= 8:
            break
    model.load_state_dict(best_state)
    report = {
        "history": history, "best_epoch": best_epoch, "best_monitor": best_monitor,
        "test": _validation_metrics(model, loaders[2], mean_t, scale_t, device),
        "training_samples": len(loaders[0].dataset), "split_scope": split_scope,
        "subsets": split_report,
    }
    _save_checkpoint(checkpoint_path, {
        "version": CHECKPOINT_VERSION, "state_dict": best_state,
        "feature_names": list(FEATURE_NAMES), "feature_units": list(FEATURE_UNITS),
        "action_names": list(ACTION_NAMES), "action_units": list(ACTION_UNITS),
        "nozzle_ids": list(map(str, records.nozzle_ids)),
        "hidden_sizes": list(HIDDEN_SIZES), "window_steps": WINDOW_STEPS, "dropout": DROPOUT,
        "feature_mean": feature_norm.mean.tolist(), "feature_scale": feature_norm.scale.tolist(),
        "label_mean": label_norm.mean.tolist(), "label_scale": label_norm.scale.tolist(),
        "source": list(map(str, np.unique(records.source))),
        "label_source": str(records.label_source), "visibility_source": str(records.visibility_source),
        "sampling_source": None if records.sampling_source is None else str(records.sampling_source),
        "config": {
            "warmup_start_lr": float(warmup_start_lr), "rmse_monitor": rmse_monitor,
            "max_gap_steps": int(max_gap_steps), "max_epochs": max_epochs, "seed": seed,
            "split_scope": split_scope,
            "feature_repair": "causal_bounded_linear",
            "target_selection": "enabled_working_domain",
            "lower": lower.tolist(), "upper": upper.tolist(),
        },
        "report": report,
    })
    return report


@dataclass(frozen=True)
class Predictor:
    model: SprayLSTM
    feature_norm: Normalisation
    label_norm: Normalisation
    nozzle_ids: tuple
    device: torch.device
    max_gap_steps: int

    def predict(self, windows):
        """Return raw next-step settings in MPa, duty fraction and degrees."""
        windows = np.asarray(windows, dtype=float)
        if (
            windows.ndim != 3 or windows.shape[1:] != (WINDOW_STEPS, len(FEATURE_NAMES))
            or not len(windows) or not np.isfinite(windows).all()
        ):
            raise ValueError("prediction windows must be finite and have shape [batch,8,25]")
        x = torch.as_tensor(
            self.feature_norm.transform(windows), dtype=torch.float32, device=self.device,
        )
        with torch.inference_mode():
            result = self.label_norm.inverse(self.model(x).cpu().numpy())
        if not np.isfinite(result).all():
            raise FloatingPointError("LSTM returned non-finite predictions")
        return result


def load_predictor(path, device="cpu"):
    """Restore a trained checkpoint and check schema before any inference."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"trained checkpoint does not exist: {path}")
    device = torch.device(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            "checkpoint is incompatible with causal window preparation and explicit enable labels; "
            "retrain with the current training schema"
        )
    for name, value in (
        ("feature_names", list(FEATURE_NAMES)),
        ("feature_units", list(FEATURE_UNITS)), ("action_names", list(ACTION_NAMES)),
        ("action_units", list(ACTION_UNITS)), ("hidden_sizes", list(HIDDEN_SIZES)),
        ("window_steps", WINDOW_STEPS), ("dropout", DROPOUT),
    ):
        if payload.get(name) != value:
            raise ValueError(f"checkpoint {name} is missing or incompatible")
    norms = []
    for prefix, length in (("feature", len(FEATURE_NAMES)), ("label", len(ACTION_NAMES))):
        normalisation = Normalisation(
            np.asarray(payload.get(f"{prefix}_mean"), dtype=float),
            np.asarray(payload.get(f"{prefix}_scale"), dtype=float),
        )
        if len(normalisation.mean) != length:
            raise ValueError(f"incompatible {prefix} normalisation")
        norms.append(normalisation)
    nozzle_ids = tuple(payload.get("nozzle_ids", ()))
    if not nozzle_ids or len(set(nozzle_ids)) != len(nozzle_ids):
        raise ValueError("checkpoint requires a unique list of nozzle IDs")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing causal feature and target configuration; retrain")
    max_gap_steps = config.get("max_gap_steps")
    if (
        type(max_gap_steps) is not int or max_gap_steps < 0
        or config.get("feature_repair") != "causal_bounded_linear"
        or config.get("target_selection") != "enabled_working_domain"
    ):
        raise ValueError("checkpoint is missing causal feature and target configuration; retrain")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not all(
        isinstance(parameter, torch.Tensor) and torch.isfinite(parameter).all()
        for parameter in state.values()
    ):
        raise ValueError("checkpoint has missing or non-finite model weights")
    model = SprayLSTM()
    model.load_state_dict(state, strict=True)
    return Predictor(model.to(device).eval(), norms[0], norms[1], nozzle_ids, device, max_gap_steps)
