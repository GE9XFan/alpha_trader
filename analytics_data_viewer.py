#!/usr/bin/env python3
"""Interactive analytics viewer for Quantisity Capital Redis data."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import redis.asyncio as redis
import yaml
from tabulate import tabulate


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import redis_keys as rkeys  # noqa: E402


def _decode(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _num(value: Any, precision: int = 2) -> str:
    if value in (None, "", "nan"):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.{precision}f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.{precision}f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.{precision}f}K"
    return f"{number:.{precision}f}"


def _percent(value: Any, precision: int = 1) -> str:
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value) * 100:.{precision}f}%"
    except (TypeError, ValueError):
        return str(value)


def _json(value: Any) -> str:
    if value is None:
        return "—"
    return json.dumps(value, indent=2, sort_keys=True, default=str)


@dataclass
class ViewerConfig:
    symbols: List[str]
    redis_host: str
    redis_port: int


class AnalyticsDataViewer:
    """Display analytics metrics in a readable layout."""

    def __init__(self, config: ViewerConfig, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AnalyticsDataViewer":
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as fh:
            config_yaml = yaml.safe_load(fh) or {}

        symbols_cfg = config_yaml.get("symbols", {})
        configured_symbols = set(symbols_cfg.get("level2", []) or [])
        configured_symbols.update(symbols_cfg.get("standard", []) or [])

        if args.symbols:
            symbols = sorted({sym.upper() for sym in args.symbols})
        else:
            symbols = sorted(symbol.upper() for symbol in configured_symbols)

        viewer_config = ViewerConfig(
            symbols=symbols,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
        )

        client = redis.Redis(
            host=viewer_config.redis_host,
            port=viewer_config.redis_port,
            decode_responses=True,
        )

        return cls(viewer_config, client)

    async def connect(self) -> bool:
        try:
            await self.redis.ping()
        except Exception as exc:  # pragma: no cover - connectivity guard
            print(f"✗ Failed to connect to Redis: {exc}")
            return False

        kwargs = self.redis.connection_pool.connection_kwargs
        print(
            f"✓ Connected to Redis at {kwargs.get('host')}:{kwargs.get('port')}\n"
        )
        return True

    async def close(self) -> None:
        await self.redis.aclose()

    async def display(self) -> None:
        if not await self.connect():
            return

        await self._display_portfolio_overview()
        await self._display_vix_snapshot()

        for symbol in self.config.symbols:
            await self._display_symbol(symbol)

        await self.close()

    async def _display_portfolio_overview(self) -> None:
        print("=" * 100)
        print("PORTFOLIO & SECTOR ANALYTICS")
        print("=" * 100)

        summary = await self._fetch_json(rkeys.analytics_portfolio_summary_key())
        if summary:
            rows = [
                ["Symbol Count", summary.get("symbol_count")],
                ["Avg VPIN", _num(summary.get("avg_vpin"), 3)],
                ["Max VPIN", _num(summary.get("max_vpin"), 3), summary.get("max_vpin_symbol")],
                ["Total GEX", _num(summary.get("total_gex"))],
                ["Total DEX", _num(summary.get("total_dex"))],
                ["Total Vanna", _num(summary.get("total_vanna_notional"))],
                ["Total Charm", _num(summary.get("total_charm_notional"))],
                ["Total Hedging", _num(summary.get("total_hedging_notional"))],
                ["Toxic Count", summary.get("toxic_count")],
            ]
            print("\n📊 Portfolio Summary:")
            print(tabulate(rows, headers=["Metric", "Value", "Notes"], tablefmt="simple"))

            sector_flows = summary.get("sector_flows") or {}
            if sector_flows:
                table = []
                for sector, payload in sorted(sector_flows.items()):
                    table.append(
                        [
                            sector,
                            _num(payload.get("total_gex")),
                            _num(payload.get("total_dex")),
                            _num(payload.get("total_vanna_notional")),
                            _num(payload.get("total_charm_notional")),
                        ]
                    )
                print("\n🏢 Sector Dealer Flow Snapshot:")
                print(
                    tabulate(
                        table,
                        headers=["Sector", "GEX", "DEX", "Vanna", "Charm"],
                        tablefmt="simple",
                    )
                )
        else:
            print("\n📊 Portfolio summary not available.")

        correlation = await self._fetch_json(rkeys.analytics_portfolio_correlation_key())
        if correlation:
            high_pairs = correlation.get("high_pairs") or []
            if high_pairs:
                table = [
                    ["-".join(pair.get("pair", [])), _num(pair.get("correlation"), 3)]
                    for pair in high_pairs
                ]
                print("\n🤝 High Correlation Pairs (|ρ| ≥ threshold):")
                print(tabulate(table, headers=["Pair", "Correlation"], tablefmt="simple"))
        else:
            print("\n📈 Correlation matrix not available.")

    async def _display_vix_snapshot(self) -> None:
        payload = await self._fetch_json(rkeys.analytics_vix1d_key())
        if not payload:
            print("\nVIX1D snapshot unavailable.\n")
            return

        print("\n" + "=" * 100)
        print("VIX1D SNAPSHOT")
        print("=" * 100)
        rows = [
            ["Value", _num(payload.get("value"))],
            ["5m Change", _num(payload.get("change_5m"))],
            ["1h Change", _num(payload.get("change_1h"))],
            ["Z-Score", _num(payload.get("zscore"), 3)],
            ["Percentile", _percent(payload.get("percentile"))],
            ["Regime", payload.get("regime", "UNKNOWN")],
            ["Samples", payload.get("samples")],
        ]
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    async def _display_symbol(self, symbol: str) -> None:
        bundle = await self._fetch_symbol_bundle(symbol)
        print("\n" + "=" * 100)
        print(f"{symbol} ANALYTICS")
        print("=" * 100)

        core_rows = []
        vpin = bundle.get("vpin") or {}
        if vpin:
            core_rows.append(["VPIN", _num(vpin.get("value"), 3), vpin.get("bucket"), vpin.get("samples")])

        gex = bundle.get("gex") or {}
        dex = bundle.get("dex") or {}
        if gex or dex:
            core_rows.extend(
                [
                    ["GEX", _num(gex.get("total_gex")), _num(gex.get("spot_gex")), gex.get("sample_count")],
                    [
                        "DEX",
                        _num(dex.get("total_dex")),
                        f"calls={_num(dex.get('call_dex'))} puts={_num(dex.get('put_dex'))}",
                        dex.get("sample_count"),
                    ],
                ]
            )

        toxicity = bundle.get("toxicity")
        if toxicity:
            toxicity_value = toxicity.get("toxicity_score", toxicity.get("vpin"))
            notes_parts: List[str] = []
            level = toxicity.get("toxicity_level")
            if level:
                notes_parts.append(level)
            vpin_value = toxicity.get("vpin")
            if vpin_value is not None:
                notes_parts.append(f"VPIN={_num(vpin_value, 3)}")
            core_rows.append([
                "Toxicity",
                _num(toxicity_value, 3),
                " ".join(notes_parts) if notes_parts else "—",
                "—",
            ])

        obi = bundle.get("obi")
        if obi:
            notes_parts = []
            state = obi.get("state")
            if state:
                notes_parts.append(state)
            pressure = obi.get("pressure_ratio")
            if pressure is not None:
                notes_parts.append(f"pressure={_num(pressure, 2)}")
            core_rows.append([
                "Order Book",
                _num(obi.get("level5_imbalance"), 3),
                " ".join(notes_parts) if notes_parts else "—",
                _num(obi.get("book_velocity"), 3),
            ])

        if core_rows:
            print("\nCore Metrics:")
            print(
                tabulate(
                    core_rows,
                    headers=["Metric", "Value", "Notes", "Samples"],
                    tablefmt="simple",
                )
            )

        self._print_dealer_flows(bundle)
        self._print_skew(bundle)
        self._print_flow_clusters(bundle)
        self._print_moc_metrics(bundle)
        self._print_unusual_activity(bundle)
        self._print_misc_metrics(bundle)

    async def _fetch_symbol_bundle(self, symbol: str) -> Dict[str, Any]:
        keys = [
            ("vpin", rkeys.analytics_vpin_key(symbol)),
            ("gex", rkeys.analytics_gex_key(symbol)),
            ("dex", rkeys.analytics_dex_key(symbol)),
            ("toxicity", rkeys.analytics_toxicity_key(symbol)),
            ("obi", rkeys.analytics_obi_key(symbol)),
            ("vanna", rkeys.analytics_vanna_key(symbol)),
            ("charm", rkeys.analytics_charm_key(symbol)),
            ("hedging", rkeys.analytics_hedging_impact_key(symbol)),
            ("skew", rkeys.analytics_skew_key(symbol)),
            ("flow_clusters", rkeys.analytics_flow_clusters_key(symbol)),
            ("sweep", rkeys.analytics_metric_key(symbol, "sweep")),
            ("hidden", rkeys.analytics_metric_key(symbol, "hidden")),
            ("gamma_pin", rkeys.analytics_metric_key(symbol, "gamma_pin")),
            ("moc", rkeys.analytics_metric_key(symbol, "moc")),
            ("unusual", rkeys.analytics_metric_key(symbol, "unusual")),
        ]

        async with self.redis.pipeline(transaction=False) as pipe:
            for _, key in keys:
                pipe.get(key)
            raw_values = await pipe.execute()

        bundle: Dict[str, Any] = {}
        for (name, _), raw in zip(keys, raw_values):
            bundle[name] = _decode(raw)
        return bundle

    async def _fetch_json(self, key: str) -> Optional[Mapping[str, Any]]:
        value = await self.redis.get(key)
        decoded = _decode(value)
        return decoded if isinstance(decoded, Mapping) else None

    def _print_dealer_flows(self, bundle: Mapping[str, Any]) -> None:
        vanna = bundle.get("vanna") or {}
        charm = bundle.get("charm") or {}
        hedging = bundle.get("hedging") or {}
        if not any([vanna, charm, hedging]):
            return

        rows = []
        if vanna:
            rows.append(
                [
                    "Vanna",
                    _num(vanna.get("total_vanna_notional_per_pct_vol")),
                    _num((vanna.get("history") or {}).get("zscore"), 2),
                    vanna.get("samples"),
                ]
            )
        if charm:
            rows.append(
                [
                    "Charm",
                    _num(charm.get("total_charm_notional_per_day")),
                    _num((charm.get("history") or {}).get("zscore"), 2),
                    charm.get("samples"),
                ]
            )
        if hedging:
            rows.append(
                [
                    "Hedging",
                    _num(hedging.get("notional_per_pct_move")),
                    _num(hedging.get("shares_per_pct_move")),
                    hedging.get("samples"),
                ]
            )

        print("\nDealer Flow Metrics:")
        print(
            tabulate(
                rows,
                headers=["Metric", "Notional", "Z / Shares", "Samples"],
                tablefmt="simple",
            )
        )

    def _print_skew(self, bundle: Mapping[str, Any]) -> None:
        skew = bundle.get("skew") or {}
        if not skew:
            return

        history = skew.get("history") or {}
        rows = [
            ["Skew", _num(skew.get("skew"), 4)],
            ["Z-Score", _num(history.get("zscore"), 2)],
            ["Samples", skew.get("samples")],
        ]
        print("\nSkew Overview:")
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    def _print_flow_clusters(self, bundle: Mapping[str, Any]) -> None:
        payload = bundle.get("flow_clusters") or {}
        if not payload:
            return

        strategy = payload.get("strategy_distribution") or {}
        participant = payload.get("participant_distribution") or {}

        strat_rows = [
            [name, _percent(weight)] for name, weight in sorted(strategy.items())
        ]
        part_rows = [
            [name, _percent(weight)] for name, weight in sorted(participant.items())
        ]

        print("\nFlow Clustering:")
        print(tabulate(strat_rows, headers=["Strategy", "Share"], tablefmt="simple"))
        print(tabulate(part_rows, headers=["Participant", "Share"], tablefmt="simple"))

    def _print_moc_metrics(self, bundle: Mapping[str, Any]) -> None:
        payload = bundle.get("moc") or {}
        if not isinstance(payload, Mapping) or not payload:
            return

        rows = [
            ["Side", payload.get("imbalance_side", "FLAT")],
            ["Imbalance", _num(payload.get("imbalance_total"))],
            ["Ratio", _percent(payload.get("imbalance_ratio"), 2)],
            ["Paired", _num(payload.get("imbalance_paired"))],
            ["Level5", _num(payload.get("level5_imbalance"), 3)],
            ["Indicative", _num(payload.get("indicative_price"))],
            ["Offset (bps)", _num(payload.get("near_close_offset_bps"), 2)],
            ["Proj Volume", _num(payload.get("projected_volume_shares"))],
            ["Time to Close (m)", _num(payload.get("minutes_to_close"), 2)],
            ["Gamma Factor", _num(payload.get("gamma_factor"), 2)],
        ]

        print("\nMOC Projection:")
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    def _print_unusual_activity(self, bundle: Mapping[str, Any]) -> None:
        payload = bundle.get("unusual") or {}
        if not isinstance(payload, Mapping) or not payload:
            return

        rows = [
            ["Score", _num(payload.get("score"), 3)],
            ["Classification", payload.get("classification", "—")],
            ["Volume/OI", _num(payload.get("volume_oi_ratio"), 3)],
            ["Max Contract Ratio", _num(payload.get("max_contract_ratio"), 3)],
            ["High Ratio Contracts", payload.get("high_ratio_contracts")],
            ["High Notional Contracts", payload.get("high_notional_contracts")],
            ["Dominant Flow", payload.get("dominant_flow", "—")],
            ["Spot", _num(payload.get("spot_price"))],
        ]

        print("\nUnusual Options Activity:")
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    def _print_misc_metrics(self, bundle: Mapping[str, Any]) -> None:
        rows = []
        if bundle.get("sweep") is not None:
            rows.append(["Sweep Score", bundle.get("sweep")])
        if bundle.get("hidden") is not None:
            rows.append(["Hidden Flow", bundle.get("hidden")])
        if bundle.get("gamma_pin") is not None:
            rows.append(["Gamma Pin", bundle.get("gamma_pin")])

        if rows:
            print("\nAdditional Metrics:")
            print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))


async def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Redis analytics viewer")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis hostname")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--symbols", nargs="*", help="Limit output to specific symbols")
    args = parser.parse_args(argv)

    viewer = AnalyticsDataViewer.from_args(args)
    await viewer.display()


if __name__ == "__main__":
    asyncio.run(main())
