#!/usr/bin/env python3
"""Discord Bot Module

Consumes tiered distribution queues and posts trade lifecycle events to
Discord via per-channel webhooks. Premium members receive full execution and
risk context immediately, while basic and free tiers get progressively
redacted summaries after their configured delays.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
import redis.asyncio as aioredis

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for <3.9 if ever needed
    from backports.zoneinfo import ZoneInfo  # type: ignore

from .discord_config import load_discord_webhooks


@dataclass
class WebhookTarget:
    key: str
    url: str
    channel_id: Optional[int]
    label: Optional[str]
    mention_role_id: Optional[int]
    mention_override: Optional[str]


@dataclass
class TierConfig:
    name: str
    queue: str
    detail_level: str
    mention: Optional[str]
    webhook: WebhookTarget

    @property
    def mention_text(self) -> str:
        parts: List[str] = []
        if self.mention:
            parts.append(self.mention)
        if self.webhook.mention_override:
            parts.append(self.webhook.mention_override)
        elif self.webhook.mention_role_id:
            parts.append(f"<@&{self.webhook.mention_role_id}>")
        return " ".join(dict.fromkeys(part for part in parts if part))  # preserve order, dedupe


@dataclass
class DiscordMessage:
    content: str
    embeds: List[Dict[str, Any]]


@dataclass
class SignalEnvelope:
    id: str
    symbol: str
    side: str
    strategy: Optional[str]
    confidence: Optional[float]
    confidence_band: Optional[str]
    action_type: str
    ts: Optional[int]
    tier: str
    execution: Dict[str, Any]
    lifecycle: Dict[str, Any]
    entry: Optional[float]
    stop: Optional[float]
    targets: List[float]
    position_notional: Optional[float]
    reasons: List[str]
    contract: Dict[str, Any]
    sentiment: Optional[str]
    teaser_message: Optional[str]
    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], tier: str) -> "SignalEnvelope":
        now_ms = int(time.time() * 1000)
        signal_id = str(data.get('id') or f"{data.get('symbol', 'NA')}:{data.get('ts', now_ms)}")
        action_type = str(data.get('action_type') or 'ENTRY').upper()
        execution = data.get('execution') or {}
        lifecycle = data.get('lifecycle') or {}
        targets_raw = data.get('targets') or []
        targets: List[float] = []
        if isinstance(targets_raw, Sequence) and not isinstance(targets_raw, (str, bytes)):
            for item in targets_raw:
                try:
                    targets.append(round(float(item), 4))
                except (TypeError, ValueError):
                    continue

        reasons = data.get('reasons') or []
        if isinstance(reasons, str):
            reasons = [reasons]
        elif not isinstance(reasons, list):
            reasons = []

        contract = data.get('contract') or {}

        return cls(
            id=signal_id,
            symbol=str(data.get('symbol') or '').upper(),
            side=str(data.get('side') or data.get('sentiment') or '').upper(),
            strategy=(data.get('strategy') or data.get('playbook')),
            confidence=_safe_float(data.get('confidence')),
            confidence_band=data.get('confidence_band'),
            action_type=action_type,
            ts=_safe_int(data.get('ts')),
            tier=tier,
            execution=execution,
            lifecycle=lifecycle,
            entry=_safe_float(data.get('entry')),
            stop=_safe_float(data.get('stop') or data.get('stop_loss')),
            targets=targets,
            position_notional=_safe_float(data.get('position_notional')),
            reasons=reasons,
            contract=contract,
            sentiment=data.get('sentiment'),
            teaser_message=data.get('message'),
            raw=data,
        )

    @property
    def executed_at(self) -> Optional[datetime]:
        executed_at = self.execution.get('executed_at') if isinstance(self.execution, dict) else None
        if executed_at:
            try:
                return datetime.fromisoformat(str(executed_at))
            except ValueError:
                pass
        if self.ts:
            try:
                return datetime.fromtimestamp(self.ts / 1000)
            except (TypeError, ValueError):
                return None
        return None

    @property
    def execution_status(self) -> str:
        status = ''
        if isinstance(self.execution, dict):
            status = str(self.execution.get('status') or '')
        return status.upper()

    @property
    def side_emoji(self) -> str:
        if self.side == 'LONG':
            return '🟢'
        if self.side == 'SHORT':
            return '🔴'
        return '🔔'

    @property
    def is_entry(self) -> bool:
        return self.action_type == 'ENTRY'

    @property
    def is_exit(self) -> bool:
        return self.action_type == 'EXIT'

    @property
    def is_scale_out(self) -> bool:
        return self.action_type == 'SCALE_OUT'


@dataclass
class AnalysisSymbol:
    symbol: str
    imbalance_total: float
    imbalance_ratio: float
    imbalance_side: str
    indicative_price: Optional[float]
    near_close_offset_bps: Optional[float]
    projected_volume_shares: Optional[float]
    gamma_factor: Optional[float]
    minutes_to_close: Optional[float]
    change_vs_yesterday: Optional[float]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisSymbol":
        def _float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        symbol = str(data.get('symbol') or '').upper()
        return cls(
            symbol=symbol,
            imbalance_total=float(data.get('imbalance_total') or 0.0),
            imbalance_ratio=float(data.get('imbalance_ratio') or 0.0),
            imbalance_side=str(data.get('imbalance_side') or 'FLAT').upper(),
            indicative_price=_float(data.get('indicative_price')),
            near_close_offset_bps=_float(data.get('near_close_offset_bps')),
            projected_volume_shares=_float(data.get('projected_volume_shares')),
            gamma_factor=_float(data.get('gamma_factor')),
            minutes_to_close=_float(data.get('minutes_to_close')),
            change_vs_yesterday=_float(data.get('change_vs_yesterday')),
        )


@dataclass
class AnalysisEnvelope:
    id: str
    payload_type: str
    status: str
    slot_label: str
    timestamp: datetime
    featured: List[AnalysisSymbol]
    watchlist: List[AnalysisSymbol]
    note: Optional[str]
    next_update_text: str
    mention_everyone: bool
    dedupe_token: str
    extreme: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisEnvelope":
        payload_type = str(data.get('type') or 'analysis.moc')
        status = str(data.get('status') or 'ok').lower()
        slot_label = str(data.get('slot_label') or '')
        timestamp_value = data.get('timestamp')

        if isinstance(timestamp_value, (int, float)):
            timestamp = datetime.fromtimestamp(float(timestamp_value), tz=timezone.utc)
        elif isinstance(timestamp_value, str):
            try:
                parsed = datetime.fromisoformat(timestamp_value)
                if parsed.tzinfo is None:
                    timestamp = parsed.replace(tzinfo=timezone.utc)
                else:
                    timestamp = parsed.astimezone(timezone.utc)
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        featured_raw = data.get('featured') or []
        watchlist_raw = data.get('watchlist') or []
        featured = [AnalysisSymbol.from_dict(item) for item in featured_raw if isinstance(item, dict)]
        watchlist = [AnalysisSymbol.from_dict(item) for item in watchlist_raw if isinstance(item, dict)]

        note = data.get('note')
        if note is not None:
            note = str(note)

        next_update = str(data.get('next_update_text') or '')
        mention_everyone = bool(data.get('mention_everyone'))
        dedupe_token = str(data.get('dedupe_token') or f"{payload_type}:{slot_label or timestamp.isoformat()}" )
        extreme = bool(data.get('extreme'))

        return cls(
            id=str(data.get('id') or f"moc:{timestamp.timestamp():.0f}"),
            payload_type=payload_type,
            status=status,
            slot_label=slot_label,
            timestamp=timestamp,
            featured=featured,
            watchlist=watchlist,
            note=note,
            next_update_text=next_update,
            mention_everyone=mention_everyone,
            dedupe_token=dedupe_token,
            extreme=extreme,
        )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


class DiscordMessageBuilder:
    """Build tier-aware Discord payloads from :class:`SignalEnvelope`."""

    def __init__(self, formatting_config: Dict[str, Any]):
        colors = formatting_config.get('colors') if formatting_config else {}
        if colors is None:
            colors = {}
        self.colors = {
            'LONG': int(colors.get('long', 0x2ECC71)),
            'SHORT': int(colors.get('short', 0xE74C3C)),
            'NEUTRAL': int(colors.get('neutral', 0x4C6EF5)),
        }
        self.premium_tags = formatting_config.get('premium_rationale_tags', {}) if formatting_config else {}
        self.basic_tags = formatting_config.get('basic_rationale_tags', {}) if formatting_config else {}
        self.basic_upgrade_text = formatting_config.get('basic_upgrade_text') if formatting_config else None
        self.free_upgrade_text = formatting_config.get('free_upgrade_text') if formatting_config else None
        self.teaser_cta = formatting_config.get('teaser_cta') if formatting_config else None

    def build(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        if tier.detail_level == 'teaser':
            return self._build_teaser_payload(envelope, tier)
        if envelope.is_exit:
            return self._build_exit_payload(envelope, tier)
        if envelope.is_scale_out:
            return self._build_scale_payload(envelope, tier)
        return self._build_entry_payload(envelope, tier)

    # ------------------------------------------------------------------
    # Entry formatting
    # ------------------------------------------------------------------
    def _build_entry_payload(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        if tier.detail_level == 'premium':
            return self._build_premium_entry(envelope, tier)
        return self._build_basic_entry(envelope, tier)

    def _build_premium_entry(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        mention = tier.mention_text or ''
        headline = f"{mention} {envelope.side_emoji} Premium fill on **{envelope.symbol}** ({envelope.strategy or 'N/A'})"
        color = self._resolve_color(envelope.side)
        confidence_text = self._format_confidence(envelope)
        fill_details = self._format_fill(envelope)
        risk_details = self._format_risk(envelope)
        reasons = self._format_reasons(envelope.reasons, self.premium_tags)

        fields = [
            {
                'name': 'Contract',
                'value': self._format_contract(envelope.contract, envelope.symbol),
                'inline': False,
            },
        ]

        if fill_details:
            fields.append({'name': 'Execution', 'value': fill_details, 'inline': False})
        if risk_details:
            fields.append({'name': 'Risk Plan', 'value': risk_details, 'inline': False})
        if confidence_text:
            fields.append({'name': 'Confidence', 'value': confidence_text, 'inline': True})
        if reasons:
            fields.append({'name': 'Context', 'value': reasons, 'inline': False})

        embed = {
            'title': f"{envelope.symbol} {envelope.side.title()} Entry",
            'color': color,
            'fields': fields,
            'timestamp': self._iso_timestamp(envelope.executed_at),
            'footer': {'text': f"Signal {envelope.id}"},
        }

        return DiscordMessage(content=headline.strip(), embeds=[embed])

    def _build_basic_entry(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        mention = (tier.mention_text or '').strip()
        confidence = self._confidence_label(envelope)
        drivers = self._format_reasons([], self.basic_tags)
        upgrade_text = self.basic_upgrade_text or 'Premium members receive instant alerts.'

        fields = [
            {'name': 'Contract', 'value': self._format_contract(envelope.contract, envelope.symbol), 'inline': False},
            {'name': 'Execution', 'value': self._format_basic_execution(envelope), 'inline': False},
        ]

        risk_plan = self._format_risk(envelope)
        if risk_plan:
            fields.append({'name': 'Risk Plan', 'value': risk_plan, 'inline': False})

        fields.append({'name': 'Confidence', 'value': confidence, 'inline': True})

        if drivers:
            fields.append({'name': 'Drivers', 'value': drivers, 'inline': False})

        fields.append({'name': 'Upgrade', 'value': upgrade_text, 'inline': False})

        embed = {
            'title': f"{envelope.symbol} {envelope.side.title()} Entry",
            'color': self._resolve_color(envelope.side),
            'fields': fields,
            'timestamp': self._iso_timestamp(envelope.executed_at),
            'footer': {'text': self._format_local_timestamp(envelope.executed_at)},
        }

        content = mention if mention else ''
        return DiscordMessage(content=content, embeds=[embed])

    def _build_teaser_payload(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        mention = (tier.mention_text or '').strip()
        upgrade_text = self.free_upgrade_text or self.teaser_cta or 'Unlock live entries, contract specs, and risk plan with Premium.'

        fields = [
            {'name': 'Contract', 'value': self._format_contract(envelope.contract, envelope.symbol), 'inline': False},
            {'name': 'Execution', 'value': self._format_teaser_execution(envelope), 'inline': False},
        ]

        risk_plan = self._format_teaser_risk(envelope)
        if risk_plan:
            fields.append({'name': 'Risk Plan', 'value': risk_plan, 'inline': False})

        embed = {
            'title': f"{envelope.symbol} {envelope.side.title()} Entry",
            'color': self._resolve_color(envelope.side if envelope.side else 'NEUTRAL'),
            'description': upgrade_text,
            'fields': fields,
            'timestamp': self._iso_timestamp(envelope.executed_at),
            'footer': {'text': self._format_local_timestamp(envelope.executed_at)},
        }

        content = mention if mention else ''
        return DiscordMessage(content=content, embeds=[embed])

    # ------------------------------------------------------------------
    # Exit formatting
    # ------------------------------------------------------------------
    def _build_exit_payload(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        if tier.detail_level == 'premium':
            return self._build_premium_exit(envelope, tier)
        return self._build_basic_exit(envelope, tier)

    def _build_premium_exit(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        mention = tier.mention_text or ''
        lifecycle = envelope.lifecycle or {}
        result = lifecycle.get('result') or envelope.execution_status or 'EXIT'
        pnl = lifecycle.get('realized_pnl')
        return_pct = lifecycle.get('return_pct')
        holding = lifecycle.get('holding_period_minutes')
        reason = lifecycle.get('reason') or envelope.raw.get('exit_reason')
        content = f"{mention} ✅ {envelope.symbol} {envelope.side.title()} closed · {result}"

        fields = []
        if pnl is not None:
            fields.append({'name': 'Realized P&L', 'value': self._format_pnl(pnl, return_pct), 'inline': True})
        if holding is not None:
            fields.append({'name': 'Holding', 'value': f"{holding} min", 'inline': True})
        if reason:
            fields.append({'name': 'Reason', 'value': str(reason), 'inline': False})

        embed = {
            'title': f"{envelope.symbol} Position Closed",
            'color': self._resolve_color('NEUTRAL'),
            'fields': fields,
            'timestamp': self._iso_timestamp(envelope.executed_at),
            'footer': {'text': f"Signal {envelope.id}"},
        }
        return DiscordMessage(content=content.strip(), embeds=[embed])

    def _build_basic_exit(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        mention = tier.mention_text or ''
        lifecycle = envelope.lifecycle or {}
        result = lifecycle.get('result') or 'Closed'
        holding = lifecycle.get('holding_period_minutes')
        content = f"{mention} ℹ️ {envelope.symbol} position closed · {result}"

        fields = []
        if result:
            fields.append({'name': 'Outcome', 'value': str(result), 'inline': True})
        if holding is not None:
            fields.append({'name': 'Holding', 'value': f"{holding} min", 'inline': True})
        upgrade_text = self.basic_upgrade_text or 'Premium members receive instant alerts.'
        fields.append({'name': 'Upgrade', 'value': upgrade_text, 'inline': False})

        embed = {
            'title': f"{envelope.symbol} Exit",
            'color': self._resolve_color('NEUTRAL'),
            'fields': fields,
            'timestamp': self._iso_timestamp(envelope.executed_at),
        }
        return DiscordMessage(content=content.strip(), embeds=[embed])

    # ------------------------------------------------------------------
    # Scale-out formatting
    # ------------------------------------------------------------------
    def _build_scale_payload(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        if tier.detail_level == 'premium':
            return self._build_premium_scale(envelope, tier)
        return self._build_basic_scale(envelope, tier)

    def _build_premium_scale(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        mention = tier.mention_text or ''
        lifecycle = envelope.lifecycle or {}
        qty = lifecycle.get('quantity')
        price = lifecycle.get('fill_price') or lifecycle.get('exit_price')
        remaining = lifecycle.get('remaining_quantity')
        reason = lifecycle.get('reason')
        content = f"{mention} ✂️ Trimmed {qty or '?'} contracts on **{envelope.symbol}**"

        description = []
        if price is not None:
            description.append(f"Fill {self._format_price(price)}")
        if remaining is not None:
            description.append(f"Remaining {remaining}")
        if reason:
            description.append(str(reason))

        embed = {
            'title': f"{envelope.symbol} Scale-Out",
            'color': self._resolve_color(envelope.side),
            'fields': [{'name': 'Details', 'value': ' · '.join(description) or 'Partial reduction executed', 'inline': False}],
            'timestamp': self._iso_timestamp(envelope.executed_at),
        }
        return DiscordMessage(content=content.strip(), embeds=[embed])

    def _build_basic_scale(self, envelope: SignalEnvelope, tier: TierConfig) -> DiscordMessage:
        mention = tier.mention_text or ''
        lifecycle = envelope.lifecycle or {}
        content = f"{mention} Position trim on **{envelope.symbol}** underway."
        remaining = lifecycle.get('remaining_quantity')
        fields = []
        if remaining is not None:
            fields.append({'name': 'Remaining', 'value': str(remaining), 'inline': True})
        upgrade_text = self.basic_upgrade_text or 'Premium members receive instant alerts.'
        fields.append({'name': 'Upgrade', 'value': upgrade_text, 'inline': False})

        embed = {
            'title': f"{envelope.symbol} Partial Close",
            'color': self._resolve_color(envelope.side),
            'fields': fields,
            'timestamp': self._iso_timestamp(envelope.executed_at),
        }
        return DiscordMessage(content=content.strip(), embeds=[embed])

    # ------------------------------------------------------------------
    # Helper formatting routines
    # ------------------------------------------------------------------
    def _resolve_color(self, side: Optional[str]) -> int:
        if side == 'LONG':
            return self.colors['LONG']
        if side == 'SHORT':
            return self.colors['SHORT']
        return self.colors['NEUTRAL']

    @staticmethod
    def _iso_timestamp(dt: Optional[datetime]) -> Optional[str]:
        if not dt:
            return None
        return dt.isoformat()

    @staticmethod
    def _human_time(dt: Optional[datetime]) -> str:
        if not dt:
            return 'time pending'
        return dt.strftime('%H:%M:%S')

    def _format_contract(self, contract: Dict[str, Any], fallback_symbol: str) -> str:
        if not contract:
            return fallback_symbol

        contract_type = str(contract.get('type') or contract.get('secType') or '').lower()
        if contract_type == 'option' or contract.get('right'):
            expiry = contract.get('expiry') or contract.get('lastTradeDateOrContractMonth')
            strike = contract.get('strike') or contract.get('strike_price')
            right = contract.get('right') or contract.get('option_type')
            parts = [fallback_symbol]
            if expiry:
                parts.append(self._format_expiry(expiry))
            if strike is not None:
                try:
                    strike_val = round(float(strike), 2)
                except (TypeError, ValueError):
                    strike_val = strike
                parts.append(f"{strike_val}{str(right or '').upper()[:1]}")
            return ' '.join(str(p) for p in parts if p)

        description = contract.get('localSymbol') or contract.get('symbol') or fallback_symbol
        exchange = contract.get('exchange') or contract.get('primaryExchange')
        if exchange:
            return f"{description} @ {exchange}"
        return str(description)

    @staticmethod
    def _format_expiry(expiry: Any) -> str:
        text = str(expiry)
        for fmt in ('%Y%m%d', '%y%m%d', '%Y-%m-%d'):
            try:
                return datetime.strptime(text, fmt).strftime('%d %b %y')
            except ValueError:
                continue
        return text

    @staticmethod
    def _format_fill(envelope: SignalEnvelope) -> Optional[str]:
        execution = envelope.execution or {}
        parts = []
        if execution.get('avg_fill_price') is not None:
            parts.append(f"Fill {DiscordMessageBuilder._format_price(execution['avg_fill_price'])}")
        if execution.get('filled_quantity') is not None:
            parts.append(f"Size {execution['filled_quantity']}")
        if envelope.position_notional is not None:
            parts.append(f"Notional {_format_currency(envelope.position_notional)}")
        return ' · '.join(parts) if parts else None

    @staticmethod
    def _format_risk(envelope: SignalEnvelope) -> Optional[str]:
        components = []
        if envelope.entry is not None:
            components.append(f"Entry {DiscordMessageBuilder._format_price(envelope.entry)}")
        if envelope.stop is not None:
            components.append(f"Stop {DiscordMessageBuilder._format_price(envelope.stop)}")
        if envelope.targets:
            components.append(f"Target {DiscordMessageBuilder._format_price(envelope.targets[0])}")
        return ' · '.join(components) if components else None

    def _format_confidence(self, envelope: SignalEnvelope) -> Optional[str]:
        if envelope.confidence is not None:
            confidence_pct = f"{round(envelope.confidence, 2)}%" if envelope.confidence > 1 else f"{round(envelope.confidence * 100, 2)}%"
            if envelope.confidence_band:
                return f"{confidence_pct} ({envelope.confidence_band})"
            return confidence_pct
        if envelope.confidence_band:
            return envelope.confidence_band
        return None

    def _confidence_label(self, envelope: SignalEnvelope) -> str:
        if envelope.confidence_band:
            return str(envelope.confidence_band).upper()

        if envelope.confidence is not None:
            raw = envelope.confidence if envelope.confidence > 1 else envelope.confidence * 100
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return 'UNSPECIFIED'

            if value >= 90:
                return 'HIGH'
            if value >= 75:
                return 'MEDIUM'
            return 'LOW'

        return 'UNSPECIFIED'

    @staticmethod
    def _format_reasons(reasons: List[str], tag_map: Dict[str, str]) -> Optional[str]:
        if reasons:
            return ' • '.join(reasons[:3])
        if tag_map:
            return ' • '.join(tag_map.values())
        return None

    @staticmethod
    def _format_basic_execution(envelope: SignalEnvelope) -> str:
        execution = envelope.execution or {}
        parts: List[str] = []
        if execution.get('avg_fill_price') is not None:
            parts.append(f"Fill {DiscordMessageBuilder._format_price(execution['avg_fill_price'])}")
        if execution.get('filled_quantity') is not None:
            parts.append(f"Size {execution['filled_quantity']}")
        return ' · '.join(parts) if parts else (envelope.execution_status or 'Pending')

    @staticmethod
    def _format_teaser_execution(envelope: SignalEnvelope) -> str:
        execution = envelope.execution or {}
        if execution.get('avg_fill_price') is not None:
            return f"Fill {DiscordMessageBuilder._format_price(execution['avg_fill_price'])}"
        return envelope.execution_status or 'Pending'

    @staticmethod
    def _format_teaser_risk(envelope: SignalEnvelope) -> Optional[str]:
        components = []
        if envelope.entry is not None:
            components.append(f"Entry {DiscordMessageBuilder._format_price(envelope.entry)}")
        if envelope.stop is not None:
            components.append(f"Stop {DiscordMessageBuilder._format_price(envelope.stop)}")
        return ' · '.join(components) if components else None

    @staticmethod
    def _format_local_timestamp(dt: Optional[datetime]) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            eastern = ZoneInfo("US/Eastern")
            local_dt = dt.astimezone(eastern)
        except Exception:  # pragma: no cover - fallback if timezone unavailable
            local_dt = dt.astimezone()

        hour = local_dt.hour % 12 or 12
        minute = local_dt.minute
        ampm = 'AM' if local_dt.hour < 12 else 'PM'
        year_suffix = local_dt.year % 100
        return f"{local_dt.month}/{local_dt.day}/{year_suffix:02d}, {hour}:{minute:02d} {ampm}"

    @staticmethod
    def _format_pnl(realized: float, return_pct: Optional[float]) -> str:
        prefix = '▲' if realized > 0 else '▼' if realized < 0 else '⏸'
        pct_text = f" ({round(return_pct * 100, 2)}%)" if return_pct is not None else ''
        return f"{prefix} {_format_currency(realized)}{pct_text}"

    @staticmethod
    def _format_price(value: Any) -> str:
        try:
            return f"${float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)


class AnalysisMessageBuilder:
    """Render Discord payloads for premium analysis drops."""

    def __init__(self, formatting_config: Dict[str, Any]):
        analysis_cfg = (formatting_config or {}).get('analysis') or {}
        color_fallback = (formatting_config or {}).get('colors') or {}
        neutral_default = int(color_fallback.get('neutral', 0x4C6EF5))
        self.neutral_color = int(analysis_cfg.get('neutral', neutral_default))
        self.extreme_color = int(analysis_cfg.get('extreme', self.neutral_color))
        try:
            self.market_tz = ZoneInfo("US/Eastern")
        except Exception:  # pragma: no cover - fallback if zoneinfo missing
            self.market_tz = timezone.utc

    def build(self, envelope: AnalysisEnvelope, tier: TierConfig) -> DiscordMessage:
        mention_parts: List[str] = []
        if envelope.mention_everyone:
            mention_parts.append('@everyone')
        elif tier.mention_text:
            mention_parts.append(tier.mention_text)

        slot_text = envelope.slot_label or self._format_slot(envelope.timestamp)
        headline = f"Premium Analysis • MOC Imbalance Update — {slot_text}"
        if mention_parts:
            content = f"{' '.join(mention_parts)} {headline}".strip()
        else:
            content = headline

        embed: Dict[str, Any] = {
            'title': self._build_title(envelope.timestamp),
            'color': self.extreme_color if envelope.extreme else self.neutral_color,
            'fields': [],
            'timestamp': envelope.timestamp.astimezone(timezone.utc).isoformat(),
            'footer': {'text': envelope.next_update_text or 'Next update timing TBA'},
        }

        if envelope.note:
            embed['description'] = envelope.note

        if envelope.status != 'ok':
            if 'description' not in embed:
                embed['description'] = 'Market close data temporarily unavailable.'
            return DiscordMessage(content=content, embeds=[embed])

        for symbol in envelope.featured:
            embed['fields'].append(self._build_feature_field(symbol))

        watchlist_field = self._build_watchlist_field(envelope.watchlist)
        if watchlist_field:
            embed['fields'].append(watchlist_field)

        if not embed['fields']:
            description = embed.get('description') or ''
            suffix = 'No qualifying imbalances detected.'
            embed['description'] = (description + ('\n' if description else '') + suffix)

        return DiscordMessage(content=content, embeds=[embed])

    def _build_title(self, timestamp: datetime) -> str:
        local_dt = timestamp.astimezone(self.market_tz)
        return f"Closing Imbalance Projection (as of {local_dt.strftime('%H:%M:%S ET')})"

    def _build_feature_field(self, symbol: AnalysisSymbol) -> Dict[str, Any]:
        side_label = symbol.imbalance_side.title()
        header = f"{symbol.symbol} — {side_label} Bias"
        lines: List[str] = []
        lines.append(
            f"• Unpaired Notional: {_format_currency(symbol.imbalance_total)} ({self._format_ratio(symbol.imbalance_ratio)})"
        )

        if symbol.near_close_offset_bps is not None:
            lines.append(f"• Indicative vs Mid: {self._format_bps(symbol.near_close_offset_bps)}")
        if symbol.projected_volume_shares is not None:
            lines.append(f"• Projected Volume: {self._format_shares(symbol.projected_volume_shares)}")
        if symbol.gamma_factor is not None:
            lines.append(f"• Gamma Factor: {symbol.gamma_factor:.2f}×")
        if symbol.minutes_to_close is not None:
            try:
                minutes = int(float(symbol.minutes_to_close))
                lines.append(f"• Minutes to Close: {minutes}")
            except (TypeError, ValueError):
                pass
        if symbol.change_vs_yesterday is not None:
            lines.append(f"• vs Yesterday: {self._format_change(symbol.change_vs_yesterday)}")

        value = '\n'.join(lines) if lines else 'No additional context available.'
        return {'name': header, 'value': value, 'inline': False}

    def _build_watchlist_field(self, symbols: List[AnalysisSymbol]) -> Optional[Dict[str, Any]]:
        if not symbols:
            return None

        entries: List[str] = []
        for symbol in symbols:
            offset = self._format_bps(symbol.near_close_offset_bps) if symbol.near_close_offset_bps is not None else '0 bps'
            entries.append(
                f"{symbol.symbol} {symbol.imbalance_side.title()} {_format_currency(symbol.imbalance_total)} ({self._format_ratio(symbol.imbalance_ratio)}) · {offset}"
            )

        value = '\n'.join(entries)
        return {'name': 'Watchlist', 'value': value, 'inline': False}

    def _format_slot(self, timestamp: datetime) -> str:
        local_dt = timestamp.astimezone(self.market_tz)
        return local_dt.strftime('%H:%M ET')

    @staticmethod
    def _format_ratio(value: float) -> str:
        return f"{value * 100:.0f}%"

    @staticmethod
    def _format_bps(value: float) -> str:
        if value is None:
            return '0 bps'
        sign = '+' if value > 0 else ''
        return f"{sign}{value:.0f} bps"

    @staticmethod
    def _format_shares(value: float) -> str:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return str(value)
        if abs(amount) >= 1_000_000_000:
            return f"{amount/1_000_000_000:.2f}B sh"
        if abs(amount) >= 1_000_000:
            return f"{amount/1_000_000:.1f}M sh"
        if abs(amount) >= 1_000:
            return f"{amount/1_000:.1f}K sh"
        return f"{amount:.0f} sh"

    @staticmethod
    def _format_change(value: float) -> str:
        sign = '+' if value > 0 else ''
        return f"{sign}{_format_currency(value)}"


def _format_currency(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:.2f}M"
    if abs(amount) >= 1_000:
        return f"${amount/1_000:.1f}K"
    return f"${amount:.2f}"


class DiscordBot:
    """Discord integration via webhooks for signal distribution."""

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)

        discord_config = config.get('discord', {})
        self.enabled = discord_config.get('enabled', False)
        self.queue_settings = discord_config.get('queue_processing', {})
        self.block_timeout = int(self.queue_settings.get('block_timeout', 2))
        self.idle_sleep = float(self.queue_settings.get('idle_sleep_seconds', 0.2))
        self.dedupe_ttl = int(self.queue_settings.get('dedupe_ttl_seconds', 900))
        self.max_retry_attempts = int(self.queue_settings.get('max_retry_attempts', 5))
        self.retry_backoff = list(self.queue_settings.get('retry_backoff_seconds', [1, 3, 7, 15, 30]))

        webhooks_config = load_discord_webhooks(discord_config.get('webhooks_file'))
        self.tiers = self._build_tier_configs(discord_config.get('tiers', {}), webhooks_config)
        self.message_builder = DiscordMessageBuilder(discord_config.get('formatting', {}))
        self.analysis_builder = AnalysisMessageBuilder(discord_config.get('formatting', {}))

        self.dead_letter_key = 'discord:dead_letter'
        self.metrics_prefix = 'discord:metrics'
        self.session: Optional[aiohttp.ClientSession] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        if not self.enabled:
            self.logger.info("Discord bot disabled in configuration")
            return
        if not self.tiers:
            self.logger.warning("Discord bot has no configured tiers; skipping start")
            return

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            tasks = [asyncio.create_task(self._run_tier_worker(tier)) for tier in self.tiers.values()]
            self.logger.info("Discord relay service started", extra={'tiers': list(self.tiers.keys())})
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                raise
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        self._stop_event.set()

    async def _run_tier_worker(self, tier: TierConfig) -> None:
        backoff_attempts = 0
        while not self._stop_event.is_set():
            try:
                result = await self.redis.brpop(tier.queue, timeout=self.block_timeout)
                if not result:
                    await asyncio.sleep(self.idle_sleep)
                    continue

                _, raw_payload = result
                if isinstance(raw_payload, bytes):
                    raw_payload = raw_payload.decode('utf-8', errors='ignore')

                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    self.logger.error("Invalid JSON payload on Discord queue", extra={'tier': tier.name})
                    await self.redis.lpush(self.dead_letter_key, raw_payload)
                    continue

                payload_type = str(payload.get('type') or '').lower()
                payload_id = str(payload.get('id') or '')

                if payload_type.startswith('analysis'):
                    analysis_envelope = AnalysisEnvelope.from_dict(payload)
                    dedupe_token = payload.get('dedupe_token') or analysis_envelope.dedupe_token
                    payload_id = analysis_envelope.id
                    message = self.analysis_builder.build(analysis_envelope, tier)
                else:
                    signal_envelope = SignalEnvelope.from_dict(payload, tier.name)
                    dedupe_token = payload.get('dedupe_token') or f"{signal_envelope.id}:{signal_envelope.action_type}"
                    payload_id = signal_envelope.id
                    message = self.message_builder.build(signal_envelope, tier)

                dedupe_key = f"discord:sent:{tier.name}:{dedupe_token}"
                if not await self.redis.setnx(dedupe_key, int(time.time())):
                    self.logger.debug(
                        "Duplicate Discord payload suppressed",
                        extra={'tier': tier.name, 'payload_type': payload_type or 'signal', 'payload_id': payload_id}
                    )
                    continue
                await self.redis.expire(dedupe_key, self.dedupe_ttl)

                await self._dispatch_message(tier, payload, message, raw_payload)

                await self.redis.incr(f"{self.metrics_prefix}:delivered:{tier.name}")
                backoff_attempts = 0

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                backoff_attempts += 1
                wait_time = self._next_backoff(backoff_attempts)
                self.logger.error(
                    "Discord worker error",
                    extra={'tier': tier.name, 'error': str(exc), 'backoff_seconds': wait_time}
                )
                await asyncio.sleep(wait_time)

    async def _dispatch_message(
        self,
        tier: TierConfig,
        _payload_meta: Dict[str, Any],
        message: DiscordMessage,
        raw_payload: str,
    ) -> None:
        if not tier.webhook.url:
            self.logger.error("Missing webhook URL", extra={'tier': tier.name})
            await self.redis.lpush(self.dead_letter_key, raw_payload)
            return

        attempts = 0
        while attempts < self.max_retry_attempts:
            attempts += 1
            try:
                async with self.session.post(
                    tier.webhook.url,
                    json={'content': message.content, 'embeds': message.embeds},
                ) as response:
                    if 200 <= response.status < 300:
                        return

                    retry_after = self._retry_after_seconds(response)
                    body = await response.text()
                    self.logger.warning(
                        "Discord webhook returned error",
                        extra={'tier': tier.name, 'status': response.status, 'body': body, 'attempt': attempts}
                    )

                    if response.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(retry_after or self._next_backoff(attempts))
                        continue

                    break  # non-retryable
            except aiohttp.ClientError as exc:
                self.logger.warning(
                    "Discord webhook transport error",
                    extra={'tier': tier.name, 'error': str(exc), 'attempt': attempts}
                )
                await asyncio.sleep(self._next_backoff(attempts))
                continue

        await self.redis.incr(f"{self.metrics_prefix}:failures:{tier.name}")
        await self.redis.lpush(self.dead_letter_key, raw_payload)

    def _build_tier_configs(
        self,
        tiers_config: Dict[str, Any],
        webhooks_config: Dict[str, Any],
    ) -> Dict[str, TierConfig]:
        tier_objects: Dict[str, TierConfig] = {}
        fallback_webhook_key = self.config.get('discord', {}).get('fallbacks', {}).get('log_webhook_key')

        for tier_name, tier_cfg in tiers_config.items():
            webhook_key = tier_cfg.get('webhook_key') or fallback_webhook_key
            webhook_data = webhooks_config.get(webhook_key or '', {}) if webhooks_config else {}
            webhook = WebhookTarget(
                key=webhook_key or tier_name,
                url=str(webhook_data.get('url') or ''),
                channel_id=_safe_int(webhook_data.get('channel_id')),
                label=webhook_data.get('label'),
                mention_role_id=_safe_int(webhook_data.get('mention_role_id')),
                mention_override=webhook_data.get('mention'),
            )

            tier_objects[tier_name] = TierConfig(
                name=tier_name,
                queue=tier_cfg.get('queue') or '',
                detail_level=str(tier_cfg.get('detail_level') or 'premium').lower(),
                mention=tier_cfg.get('mention'),
                webhook=webhook,
            )

        return {name: cfg for name, cfg in tier_objects.items() if cfg.queue and cfg.webhook.url}

    def _next_backoff(self, attempt: int) -> float:
        if attempt <= 0:
            return self.idle_sleep
        index = min(attempt - 1, len(self.retry_backoff) - 1)
        return float(self.retry_backoff[index])

    @staticmethod
    def _retry_after_seconds(response: aiohttp.ClientResponse) -> Optional[float]:
        retry_after = response.headers.get('Retry-After') if hasattr(response, 'headers') else None
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            return None


__all__ = [
    'DiscordBot',
    'DiscordMessageBuilder',
    'AnalysisMessageBuilder',
    'AnalysisEnvelope',
    'AnalysisSymbol',
    'SignalEnvelope',
    'TierConfig',
    'WebhookTarget',
]
