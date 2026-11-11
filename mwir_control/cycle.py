"""Causal cycle state, actuator boundary and observation-matched replay."""

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional, Protocol, Tuple

import numpy as np

from .control import Selection, select_action
from .data import repair_causal_window
from .nozzles import Action


SECOND_NS = 1_000_000_000
WINDOW_LENGTH = 8


@dataclass(frozen=True)
class CycleKey:
    record_id: str
    timestamp_ns: int

    def __post_init__(self):
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id must identify a continuous record")
        if not isinstance(self.timestamp_ns, (int, np.integer)) or isinstance(self.timestamp_ns, bool):
            raise ValueError("timestamp_ns must be integer nanoseconds on the record time base")

    def successor(self):
        return CycleKey(self.record_id, int(self.timestamp_ns) + SECOND_NS)


@dataclass(frozen=True)
class Observation:
    key: CycleKey
    frames: np.ndarray
    valid: np.ndarray
    context: Any
    source: str
    acquisition_ms: Optional[float] = None


@dataclass(frozen=True)
class ActualReceipt:
    key: CycleKey
    nozzle_ids: Tuple[str, ...]
    action: Action
    source: str


@dataclass(frozen=True, eq=False)
class EnablePlan:
    """External nozzle availability known by the cycle that issues the plan."""

    key: CycleKey
    nozzle_ids: Tuple[str, ...]
    enabled: np.ndarray
    source: str
    available_timestamp_ns: int

    def __post_init__(self):
        if not isinstance(self.key, CycleKey):
            raise ValueError("Enable plan key must be a CycleKey")
        ids = tuple(self.nozzle_ids)
        if (not ids or len(set(ids)) != len(ids)
                or any(not isinstance(value, str) or not value.strip() for value in ids)):
            raise ValueError("Enable plan requires unique nonempty nozzle IDs")
        enabled = np.array(self.enabled, copy=True)
        if enabled.shape != (len(ids),) or enabled.dtype.kind != "b":
            raise ValueError("Enable plan enabled must be Boolean [N] in nozzle_ids order")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Enable plan source must be a nonempty identifier")
        available = self.available_timestamp_ns
        if (not isinstance(available, (int, np.integer)) or isinstance(available, (bool, np.bool_))
                or available >= self.key.timestamp_ns):
            raise ValueError("Enable plan available_timestamp_ns must be an integer before its target timestamp")
        enabled.setflags(write=False)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "nozzle_ids", ids)
        object.__setattr__(self, "available_timestamp_ns", int(available))

    def matches(self, other):
        return (
            self.key == other.key and self.nozzle_ids == other.nozzle_ids
            and self.source == other.source
            and self.available_timestamp_ns == other.available_timestamp_ns
            and np.array_equal(self.enabled, other.enabled)
        )


@dataclass(frozen=True)
class Feedback:
    key: CycleKey
    observation_source: str
    actual: ActualReceipt
    modelled: np.ndarray
    residual: np.ndarray
    features: np.ndarray
    physical_ms: float
    preprocessing_ms: float
    acquisition_ms: Optional[float]


@dataclass(frozen=True)
class Plan:
    issued_from: CycleKey
    target: CycleKey
    nozzle_ids: Tuple[str, ...]
    selection: Selection
    inference_ms: float
    selection_ms: float
    window_ms: float
    enable_plan: EnablePlan


@dataclass(frozen=True)
class CommandAck:
    """Transport acknowledgement. Actual settings arrive in ActualReceipt."""

    target: CycleKey
    accepted: bool
    reference: str


class Actuator(Protocol):
    def issue(self, plan: Plan) -> CommandAck:
        ...


@dataclass(frozen=True)
class Dispatch:
    ack: CommandAck
    transmission_ms: float


def _elapsed_ms(start):
    return (perf_counter() - start) * 1000


class CycleController:
    """Pair measured actions with observations before constructing a plan.

    The physical evaluator receives (Action, context); the score callback
    receives (frames, context, Action). Both use the latest acquired boundary for
    the virtual next-cycle calculation. All state lives within one session.
    """

    def __init__(self, registry, evaluate, make_features, predict, score, *, max_gap_steps=0):
        if (not isinstance(max_gap_steps, (int, np.integer))
                or isinstance(max_gap_steps, (bool, np.bool_)) or max_gap_steps < 0):
            raise ValueError("max_gap_steps must be a non-negative integer")
        self.registry = registry
        self.evaluate = evaluate
        self.make_features = make_features
        self.predict = predict
        self.score = score
        self.max_gap_steps = int(max_gap_steps)
        self._history = deque(maxlen=WINDOW_LENGTH + self.max_gap_steps)
        self._latest = None
        self._plan = None
        self._issue_attempted = False
        self._seen_records = set()
        self.proposal_status = "no_observation"

    def ingest(self, observation, receipt):
        """Create the Eq. (18) residual for this recorded actual setting."""
        key = observation.key
        if receipt.key != key:
            raise ValueError("Actual action and observation must have the same record and timestamp")
        if tuple(receipt.nozzle_ids) != self.registry.ids:
            raise ValueError("Actual receipt nozzle IDs do not match registry ordering")
        if not observation.source or not receipt.source:
            raise ValueError("Observation and actual receipt require source identifiers")
        acquisition = observation.acquisition_ms
        if acquisition is not None and (not np.isfinite(acquisition) or acquisition < 0):
            raise ValueError("acquisition_ms must be finite and non-negative")
        self.registry.validate_action(receipt.action)
        contiguous = False
        if self._latest is not None:
            previous = self._latest.key
            if key.record_id == previous.record_id:
                delta = int(key.timestamp_ns) - int(previous.timestamp_ns)
                if delta <= 0 or delta % SECOND_NS:
                    raise ValueError("Cycle timestamps must increase on the record's 1 Hz grid")
                contiguous = delta == SECOND_NS
            elif key.record_id in self._seen_records:
                raise ValueError("Continuous records must be processed in contiguous blocks")

        frames = np.asarray(observation.frames, dtype=float)
        valid = np.asarray(observation.valid)
        if frames.ndim != 3 or frames.shape[0] != 2:
            raise ValueError("Observed frames must have shape [2,H,W]")
        if valid.dtype.kind != "b" or valid.shape not in (frames.shape[1:], frames.shape):
            raise ValueError("Observed validity must be a Boolean [H,W] or [2,H,W] array")
        if np.any(np.isinf(frames)):
            raise ValueError("Observation contains infinite radiance")
        start = perf_counter()
        modelled = np.array(self.evaluate(receipt.action, observation.context), dtype=float, copy=True)
        physical_ms = _elapsed_ms(start)
        if modelled.shape != frames.shape or np.any(np.isinf(modelled)):
            raise ValueError("Physical frames must be co-registered [2,H,W] and contain no infinity")
        start = perf_counter()
        support = valid & np.isfinite(frames) & np.isfinite(modelled)
        measured = np.where(support, frames, np.nan)
        modelled[~support] = np.nan
        residual = measured - modelled
        features = np.array(
            self.make_features(measured, modelled, receipt.action, observation.context),
            dtype=float, copy=True,
        )
        if features.shape != (len(self.registry), 25) or np.isinf(features).any():
            raise ValueError("Feature builder must return [N,25] with missing values marked by NaN")
        for array in (modelled, residual, features):
            array.setflags(write=False)
        if not contiguous:
            self._history.clear()
        self._history.append(features)
        self._latest = observation
        self._seen_records.add(key.record_id)
        self._plan = None
        self._issue_attempted = False
        self.proposal_status = "not_requested"
        return Feedback(
            key, observation.source, receipt, modelled, residual, features,
            physical_ms, _elapsed_ms(start), acquisition,
        )

    def propose(self, enable_plan):
        """Evaluate a next-cycle plan under its externally supplied enable mask."""
        if self._latest is None:
            self.proposal_status = "no_observation"
            return None
        if enable_plan is None:
            self.proposal_status = "missing_enable_plan"
            return None
        target = self._latest.key.successor()
        if not isinstance(enable_plan, EnablePlan) or enable_plan.key != target:
            raise ValueError("Enable plan must identify the next cycle of the current record")
        if enable_plan.nozzle_ids != self.registry.ids:
            raise ValueError("Enable plan nozzle IDs do not match registry ordering")
        if enable_plan.available_timestamp_ns > self._latest.key.timestamp_ns:
            raise ValueError("Enable plan was unavailable at the issuing observation timestamp")
        enabled = self.registry.validate_enabled(enable_plan.enabled)
        if self._plan is not None:
            if not self._plan.enable_plan.matches(enable_plan):
                raise ValueError("An action has already been selected for a different enable plan at this target")
            self.proposal_status = "planned"
            return self._plan
        raw = None
        window_ms = inference_ms = 0.0
        if enabled.any():
            if len(self._history) < WINDOW_LENGTH:
                self.proposal_status = "insufficient_history"
                return None
            start = perf_counter()
            history = np.stack(tuple(self._history))[:, enabled, :]
            windows = repair_causal_window(history, self.max_gap_steps, WINDOW_LENGTH).transpose(1, 0, 2)
            if not np.isfinite(windows).all():
                self.proposal_status = "invalid_enabled_windows"
                return None
            window_ms = _elapsed_ms(start)
            start = perf_counter()
            prediction = np.asarray(self.predict(windows), dtype=float)
            if prediction.shape != (int(enabled.sum()), 3):
                raise ValueError("Predictor must return one [MPa, duty, deg] row per enabled nozzle")
            raw = np.full((len(self.registry), 3), np.nan)
            raw[enabled] = prediction
            inference_ms = _elapsed_ms(start)
        start = perf_counter()
        context = self._latest.context

        def candidate(action):
            return self.score(self.evaluate(action, context), context, action)

        selection = select_action(raw, self.registry, candidate, enabled=enabled)
        self._plan = Plan(
            self._latest.key, target, self.registry.ids,
            selection, inference_ms, _elapsed_ms(start), window_ms, enable_plan,
        )
        self.proposal_status = "planned"
        return self._plan

    def issue(self, plan, actuator):
        """Transmit once. Reconciliation after an uncertain ACK is external."""
        if plan is None or plan is not self._plan:
            raise ValueError("Only the latest cycle's selected plan can be transmitted")
        if self._issue_attempted:
            raise ValueError("Transmission already attempted for this plan")
        self._issue_attempted = True
        start = perf_counter()
        ack = actuator.issue(plan)
        duration = _elapsed_ms(start)
        if not isinstance(ack, CommandAck) or ack.target != plan.target or not ack.reference:
            raise ValueError("Actuator acknowledgement must identify the targeted cycle")
        return Dispatch(ack, duration)


def replay(controller, pairs, enable_plans):
    """Replay recorded feedback and assess explicitly scheduled virtual actions."""
    for observation, actual in pairs:
        feedback = controller.ingest(observation, actual)
        enable_plan = enable_plans.get(observation.key.successor())
        yield feedback, controller.propose(enable_plan)


def cycle_log(feedback, plan=None, dispatch=None, *, requested_enable=None, proposal_status=None):
    """JSON-ready provenance and measured compute timing for one cycle."""
    entry = {
        "record_id": feedback.key.record_id,
        "timestamp_ns": int(feedback.key.timestamp_ns),
        "observation_source": feedback.observation_source,
        "actual_source": feedback.actual.source,
        "nozzle_ids": list(feedback.actual.nozzle_ids),
        "actual": feedback.actual.action.values.tolist(),
        "actual_enabled": feedback.actual.action.enabled.tolist(),
        "physical_ms": feedback.physical_ms,
        "preprocessing_ms": feedback.preprocessing_ms,
        "acquisition_ms": feedback.acquisition_ms,
        "actual_evaluations": 1,
    }
    if proposal_status is not None:
        entry["proposal_status"] = proposal_status
    if requested_enable is not None:
        if requested_enable.key != feedback.key.successor() or requested_enable.nozzle_ids != tuple(feedback.actual.nozzle_ids):
            raise ValueError("Requested enable plan must match the next cycle and registry order")
        entry["requested_enable"] = {
            "target_timestamp_ns": int(requested_enable.key.timestamp_ns),
            "available_timestamp_ns": requested_enable.available_timestamp_ns,
            "source": requested_enable.source,
            "enabled": requested_enable.enabled.tolist(),
        }
    if plan is not None:
        if plan.issued_from != feedback.key:
            raise ValueError("Selected plan must have been issued from the feedback cycle")
        if requested_enable is not None and not plan.enable_plan.matches(requested_enable):
            raise ValueError("Selected plan uses a different enable plan from the logged request")
        result = plan.selection
        raw = result.raw
        entry["plan"] = {
            "target_timestamp_ns": int(plan.target.timestamp_ns),
            "enable_source": plan.enable_plan.source,
            "enable_available_timestamp_ns": plan.enable_plan.available_timestamp_ns,
            "raw": None if raw is None else [row.tolist() if np.isfinite(row).all() else None for row in raw],
            "projected": result.projected.values.tolist(),
            "projected_enabled": result.projected.enabled.tolist(),
            "final": result.final.values.tolist(),
            "final_enabled": result.final.enabled.tolist(),
            "projected_scores": result.projected_scores,
            "final_scores": result.final_scores,
            "vso": [
                {
                    "nozzle_id": search.nozzle_id, "region": search.region,
                    "iterations": search.iterations, "reason": search.reason,
                    "iterates": [vars(candidate) for candidate in search.iterates],
                    "probes": [vars(candidate) for candidate in search.probes],
                    "best": vars(search.best), "selected": vars(search.selected),
                }
                for search in result.searches
            ],
            "virtual_evaluations": result.evaluations,
            "inference_ms": plan.inference_ms,
            "selection_ms": plan.selection_ms,
            "window_ms": plan.window_ms,
            "projected_evaluation_ms": result.projected_evaluation_ms,
            "guard_ms": result.guard_ms,
            "vso_ms": result.vso_ms,
        }
    if dispatch is not None:
        if plan is None or dispatch.ack.target != plan.target:
            raise ValueError("Dispatch must belong to the supplied plan")
        entry["transmission"] = {
            "accepted": dispatch.ack.accepted,
            "reference": dispatch.ack.reference,
            "transmission_ms": dispatch.transmission_ms,
        }
        if feedback.acquisition_ms is not None:
            stages = {
                "io": feedback.acquisition_ms,
                "phys": feedback.physical_ms + result.projected_evaluation_ms,
                "prep": feedback.preprocessing_ms + plan.window_ms,
                "lstm": plan.inference_ms,
                "vso": result.vso_ms,
                "guard": result.guard_ms,
                "tx": dispatch.transmission_ms,
            }
            entry["latency_stages_ms"] = stages
            entry["end_to_end_ms"] = sum(stages.values())
    return entry
