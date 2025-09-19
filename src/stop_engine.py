"""Dynamic stop-loss engine for AlphaTraderPro.

This module mirrors the supplied reference stop manager by loading detailed
profit milestones from configuration, enforcing one-way ratcheting, and
producing stop/target instructions that downstream execution components can
apply immediately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class StopStage:
    """Single profit milestone definition."""

    profit_pct: float
    stop_type: str
    stop_value: float
    take_fraction: Optional[float] = None
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StopStage":
        if "stop_type" not in data or "stop_value" not in data:
            raise ValueError("Stop stage requires 'stop_type' and 'stop_value'")
        take_fraction = data.get("take_fraction")
        return cls(
            profit_pct=float(data.get("profit_pct", 0.0)),
            stop_type=str(data["stop_type"]).upper(),
            stop_value=float(data["stop_value"]),
            take_fraction=float(take_fraction) if take_fraction is not None else None,
            name=str(data.get("name")) if data.get("name") else None,
        )


@dataclass
class StopProfile:
    """Dynamic stop configuration for a specific instrument type."""

    instrument: str
    initial_risk_pct: float
    min_tick: float
    min_improvement_pct: float
    stages: List[StopStage] = field(default_factory=list)
    hard_floor_pct: Optional[float] = None
    hard_cap_pct: Optional[float] = None

    def __post_init__(self) -> None:
        self.stages = sorted(self.stages, key=lambda stage: stage.profit_pct)

    def stage_for(self, profit_pct: float) -> Optional[StopStage]:
        """Return the last stage achieved for the supplied profit percentage."""
        stage: Optional[StopStage] = None
        for candidate in self.stages:
            if profit_pct + 1e-9 >= candidate.profit_pct:
                stage = candidate
            else:
                break
        return stage


@dataclass
class TargetAllocation:
    profit_pct: float
    fraction: float
    price: float
    quantity: int


@dataclass
class StopState:
    """Mutable stop state tracked per position."""

    highest_profit_pct: float = 0.0
    highest_milestone: float = 0.0
    last_stop_price: Optional[float] = None
    last_stop_type: Optional[str] = None
    last_stop_value: Optional[float] = None
    last_reason: Optional[str] = None
    last_update_ts: float = field(default_factory=time.time)
    filled_targets: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "StopState":
        if not data:
            return cls()
        return cls(
            highest_profit_pct=float(data.get("highest_profit_pct", 0.0)),
            highest_milestone=float(data.get("highest_milestone", 0.0)),
            last_stop_price=data.get("last_stop_price"),
            last_stop_type=data.get("last_stop_type"),
            last_stop_value=data.get("last_stop_value"),
            last_reason=data.get("last_reason"),
            last_update_ts=float(data.get("last_update_ts", time.time())),
            filled_targets=list(data.get("filled_targets", [])),
        )


@dataclass
class StopDecision:
    """Result produced when evaluating whether a stop should move."""

    should_update: bool
    stop_price: float
    reason: str
    state: StopState
    profit_pct: float
    improvement: float
    stop_type: Optional[str] = None
    stop_value: Optional[float] = None


# ---------------------------------------------------------------------------
# Stop engine implementation
# ---------------------------------------------------------------------------


class StopEngine:
    """Compute adaptive stop prices from configuration-driven rules."""

    PROFILE_FILENAME = "stop_profiles.yaml"

    def __init__(self, profiles: Dict[str, StopProfile]):
        if "default" not in profiles:
            raise ValueError("Stop profile configuration must include a 'default' profile")
        self._profiles = profiles

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_risk_config(cls, risk_config: Optional[Dict[str, Any]]) -> "StopEngine":
        profiles = cls._load_profiles()
        overrides = (risk_config or {}).get("dynamic_stops", {})
        for name, cfg in overrides.items():
            if not isinstance(cfg, dict):
                continue
            key = name.lower()
            profile = profiles.get(key)
            if not profile:
                continue
            if "initial_risk_pct" in cfg:
                profile.initial_risk_pct = float(cfg["initial_risk_pct"])
            if "min_tick" in cfg:
                profile.min_tick = float(cfg["min_tick"])
            if "min_improvement_pct" in cfg:
                profile.min_improvement_pct = float(cfg["min_improvement_pct"])
            if "stages" in cfg:
                profile.stages = [StopStage.from_dict(stage) for stage in cfg.get("stages", [])]
                profile.__post_init__()
        return cls(profiles)

    @classmethod
    def _profile_path(cls) -> Path:
        return Path(__file__).resolve().parent.parent / "config" / cls.PROFILE_FILENAME

    @classmethod
    def _load_profiles(cls) -> Dict[str, StopProfile]:
        path = cls._profile_path()
        if not path.exists():
            raise FileNotFoundError(f"Stop profile configuration not found at {path}")
        raw_profiles = yaml.safe_load(path.read_text()) or {}
        profiles: Dict[str, StopProfile] = {}
        for name, cfg in raw_profiles.items():
            stages = [StopStage.from_dict(stage) for stage in cfg.get("stages", [])]
            profiles[name.lower()] = StopProfile(
                instrument=name.lower(),
                initial_risk_pct=float(cfg.get("initial_risk_pct", 0.2)),
                min_tick=float(cfg.get("min_tick", 0.01)),
                min_improvement_pct=float(cfg.get("min_improvement_pct", 0.01)),
                stages=stages,
                hard_floor_pct=float(cfg["hard_floor_pct"]) if cfg.get("hard_floor_pct") is not None else None,
                hard_cap_pct=float(cfg["hard_cap_pct"]) if cfg.get("hard_cap_pct") is not None else None,
            )
        if "default" not in profiles:
            raise ValueError("Stop profile configuration must include a 'default' profile")
        return profiles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def profile_for(self, instrument: str) -> StopProfile:
        key = (instrument or "").lower()
        if key in {"stk", "stock"}:
            key = "stock"
        elif key in {"opt", "option", "options"}:
            key = "option"
        return self._profiles.get(key) or self._profiles["default"]

    @staticmethod
    def target_price(side: str, entry_price: float, profit_pct: float) -> float:
        if profit_pct is None:
            return entry_price
        if side == "LONG":
            return entry_price * (1 + profit_pct / 100.0)
        return entry_price * (1 - profit_pct / 100.0)

    def quantize_price(self, instrument: str, side: str, price: float) -> float:
        profile = self.profile_for(instrument)
        return self._apply_tick(price, profile.min_tick, (side or "").upper())

    def initial_stop(
        self,
        instrument: str,
        side: str,
        entry_price: float,
        *,
        state: Optional[StopState] = None,
    ) -> Tuple[float, StopState]:
        profile = self.profile_for(instrument)
        normalized_side = (side or "").upper()
        stop_state = state or StopState()
        price, stop_type, stop_value = self._initial_stop_components(profile, normalized_side, entry_price)
        stop_state.last_stop_price = price
        stop_state.last_stop_type = stop_type
        stop_state.last_stop_value = stop_value
        stop_state.last_reason = "initial_risk"
        stop_state.last_update_ts = time.time()
        stop_state.highest_profit_pct = max(0.0, stop_state.highest_profit_pct)
        return price, stop_state

    def evaluate(
        self,
        instrument: str,
        side: str,
        entry_price: float,
        current_price: Optional[float],
        *,
        current_stop: Optional[float] = None,
        state: Optional[StopState] = None,
    ) -> Optional[StopDecision]:
        if entry_price <= 0 or current_price is None or current_price <= 0:
            return None

        profile = self.profile_for(instrument)
        normalized_side = (side or "").upper()
        if normalized_side not in {"LONG", "SHORT"}:
            return None

        stop_state = state or StopState()
        profit_pct = self._profit_pct(normalized_side, entry_price, current_price)
        stop_state.highest_profit_pct = max(stop_state.highest_profit_pct, profit_pct)

        stage = profile.stage_for(profit_pct)
        if stage:
            stop_state.highest_milestone = max(stop_state.highest_milestone, stage.profit_pct)
        target_stage = profile.stage_for(stop_state.highest_milestone) or profile.stage_for(0.0)

        if target_stage:
            candidate_price = self._price_from_stage(
                target_stage,
                normalized_side,
                entry_price,
                current_price,
                stop_state,
            )
            stop_type = target_stage.stop_type
            stop_value = target_stage.stop_value
            reason = target_stage.name or f"milestone_{target_stage.profit_pct:g}"
        else:
            candidate_price = None
            stop_type = None
            stop_value = None
            reason = "initial_risk"

        initial_price, initial_type, initial_value = self._initial_stop_components(profile, normalized_side, entry_price)

        if stop_state.last_stop_price is None:
            if candidate_price is None or self._is_looser(normalized_side, candidate_price, initial_price):
                candidate_price = initial_price
                stop_type = initial_type
                stop_value = initial_value
                reason = "initial_risk"
        elif candidate_price is not None and self._is_looser(normalized_side, candidate_price, stop_state.last_stop_price):
            candidate_price = stop_state.last_stop_price
            stop_type = stop_state.last_stop_type
            stop_value = stop_state.last_stop_value
            reason = stop_state.last_reason or reason

        if candidate_price is None:
            return None

        candidate_price = self._apply_tick(candidate_price, profile.min_tick, normalized_side)

        improvement = self._improvement(
            normalized_side,
            candidate_price,
            current_stop,
            self._min_move(profile, entry_price),
        )
        should_update = current_stop is None or improvement > 0

        if should_update:
            stop_state.last_stop_price = candidate_price
            stop_state.last_stop_type = stop_type
            stop_state.last_stop_value = stop_value
            stop_state.last_reason = reason
            stop_state.last_update_ts = time.time()

        return StopDecision(
            should_update=should_update,
            stop_price=round(candidate_price, 4),
            reason=reason,
            state=stop_state,
            profit_pct=profit_pct,
            improvement=improvement,
            stop_type=stop_type,
            stop_value=stop_value,
        )

    # ------------------------------------------------------------------
    # Target planning helpers
    # ------------------------------------------------------------------
    def plan_targets(
        self,
        instrument: str,
        side: str,
        entry_price: float,
        quantity: int,
        *,
        filled_targets: Optional[Iterable[float]] = None,
    ) -> List[TargetAllocation]:
        if quantity <= 0:
            return []

        profile = self.profile_for(instrument)
        normalized_side = (side or "").upper()
        if normalized_side not in {"LONG", "SHORT"}:
            return []

        stages = [stage for stage in profile.stages if stage.take_fraction]
        if not stages:
            return []

        filled = {round(float(p), 6) for p in (filled_targets or [])}
        stages = [stage for stage in stages if round(stage.profit_pct, 6) not in filled]
        if not stages:
            return []

        is_option = instrument and instrument.lower() in {"opt", "option", "options"}
        if is_option and quantity < len(stages):
            stages = stages[:quantity]

        total_fraction = sum(max(stage.take_fraction or 0.0, 0.0) for stage in stages)
        if total_fraction <= 0:
            return []

        remaining_qty = quantity
        remaining_fraction = total_fraction
        allocations: List[TargetAllocation] = []

        for idx, stage in enumerate(stages):
            fraction = max(stage.take_fraction or 0.0, 0.0)
            if fraction <= 0:
                continue

            if idx == len(stages) - 1:
                stage_qty = remaining_qty
            else:
                stage_qty = int(round(quantity * (fraction / remaining_fraction)))
                stage_qty = max(1, min(stage_qty, remaining_qty - (len(stages) - idx - 1)))

            stage_qty = min(stage_qty, remaining_qty)
            if stage_qty <= 0:
                continue

            price = self.target_price(normalized_side, entry_price, stage.profit_pct)
            price = round(self._apply_tick(price, profile.min_tick, normalized_side), 4)
            allocations.append(
                TargetAllocation(
                    profit_pct=stage.profit_pct,
                    fraction=fraction / total_fraction,
                    price=price,
                    quantity=stage_qty,
                )
            )
            remaining_qty -= stage_qty
            remaining_fraction -= fraction

        if allocations and remaining_qty > 0:
            allocations[-1].quantity += remaining_qty

        return allocations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _profit_pct(side: str, entry: float, current: float) -> float:
        if side == "LONG":
            return (current - entry) / entry
        return (entry - current) / entry

    def _initial_stop_components(self, profile: StopProfile, side: str, entry: float) -> Tuple[float, str, float]:
        risk_pct = max(profile.initial_risk_pct, 0.0)
        percent = risk_pct * 100.0 if risk_pct <= 1 else risk_pct
        if side == "LONG":
            price = entry * (1 - percent / 100.0)
        else:
            price = entry * (1 + percent / 100.0)
        price = self._apply_tick(price, profile.min_tick, side)
        return price, "FIXED_PERCENT", percent

    def _price_from_stage(
        self,
        stage: StopStage,
        side: str,
        entry_price: float,
        current_price: float,
        state: StopState,
    ) -> float:
        pct = stage.stop_value
        if stage.stop_type == "TRAIL_PERCENT":
            if side == "LONG":
                peak = entry_price * (1 + max(state.highest_profit_pct, 0.0))
                reference = max(current_price, peak)
                price = reference * (1 - pct / 100.0)
            else:
                peak = entry_price * (1 - max(state.highest_profit_pct, 0.0))
                reference = min(current_price, peak)
                price = reference * (1 + pct / 100.0)
        elif stage.stop_type == "FIXED_PERCENT":
            if side == "LONG":
                price = entry_price * (1 - pct / 100.0)
            else:
                price = entry_price * (1 + pct / 100.0)
        else:
            raise ValueError(f"Unsupported stop_type '{stage.stop_type}'")
        return price

    @staticmethod
    def _apply_tick(price: float, min_tick: float, side: str) -> float:
        if price is None or min_tick <= 0:
            return price
        ticks = round(price / min_tick)
        quantised = ticks * min_tick
        if side == "LONG" and quantised < price:
            quantised += min_tick
        elif side == "SHORT" and quantised > price:
            quantised -= min_tick
        return quantised

    @staticmethod
    def _is_looser(side: str, candidate: float, reference: Optional[float]) -> bool:
        if candidate is None or reference is None:
            return False
        if side == "LONG":
            return candidate <= reference + 1e-6
        return candidate >= reference - 1e-6

    @staticmethod
    def _min_move(profile: StopProfile, entry_price: float) -> float:
        return max(profile.min_tick, abs(entry_price) * profile.min_improvement_pct)

    @staticmethod
    def _improvement(side: str, candidate: float, current_stop: Optional[float], min_move: float) -> float:
        if candidate is None or current_stop is None:
            return abs(candidate or 0.0)
        if side == "LONG":
            delta = candidate - current_stop
        else:
            delta = current_stop - candidate
        return delta if delta > min_move else 0.0


__all__ = [
    "StopDecision",
    "StopEngine",
    "StopProfile",
    "StopStage",
    "StopState",
    "TargetAllocation",
]
