#!/usr/bin/env python3
"""
Redis Data Viewer - Displays trading system data in a readable format
"""
import asyncio
import json
import redis.asyncio as redis
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional
from tabulate import tabulate
import sys

class RedisDataViewer:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    async def connect(self):
        """Test Redis connection"""
        try:
            await self.redis.ping()
            print(f"✓ Connected to Redis at {self.redis.connection_pool.connection_kwargs['host']}:{self.redis.connection_pool.connection_kwargs['port']}\n")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Redis: {e}")
            return False

    async def get_portfolio_analytics(self):
        """Display portfolio and sector analytics"""
        print("\n" + "="*80)
        print("PORTFOLIO & SECTOR ANALYTICS")
        print("="*80)

        # Portfolio Summary
        summary = await self.redis.get('analytics:portfolio:summary')
        if summary:
            print("\n📊 Portfolio Summary:")
            self._print_json(summary)
        else:
            print("\n📊 Portfolio Summary: No data")

        # Portfolio Correlation
        correlation = await self.redis.get('analytics:portfolio:correlation')
        if correlation:
            print("\n📈 Portfolio Correlation Matrix:")
            self._print_json(correlation)
        else:
            print("\n📈 Portfolio Correlation: No data")

        # Sector Analytics (scan for all sectors)
        sector_keys = await self.redis.keys('analytics:sector:*')
        if sector_keys:
            print("\n🏢 Sector Analytics:")
            for key in sector_keys:
                sector = key.split(':')[-1]
                data = await self.redis.get(key)
                if data:
                    print(f"\n  Sector: {sector}")
                    self._print_json(data, indent=4)
        else:
            print("\n🏢 No sector data available")

    async def get_signals(self):
        """Display signal queues and data"""
        print("\n" + "="*80)
        print("SIGNALS")
        print("="*80)

        # Pending signals queue
        pending_len = await self.redis.llen('signals:pending')
        print(f"\n📥 Pending Signals Queue: {pending_len} items")
        if pending_len > 0:
            signals = await self.redis.lrange('signals:pending', 0, min(5, pending_len-1))
            for i, signal in enumerate(signals[:5], 1):
                print(f"\n  Signal {i}:")
                self._print_json(signal, indent=4)
            if pending_len > 5:
                print(f"\n  ... and {pending_len - 5} more")

        # Execution queue
        exec_len = await self.redis.llen('signals:execution')
        print(f"\n⚡ Execution Queue: {exec_len} items")
        if exec_len > 0:
            signals = await self.redis.lrange('signals:execution', 0, min(5, exec_len-1))
            for i, signal in enumerate(signals[:5], 1):
                print(f"\n  Signal {i}:")
                self._print_json(signal, indent=4)

        # Latest signals per symbol
        latest_keys = await self.redis.keys('signals:latest:*')
        if latest_keys:
            print(f"\n📌 Latest Signals ({len(latest_keys)} symbols):")
            for key in latest_keys[:10]:
                symbol = key.split(':')[-1]
                data = await self.redis.get(key)
                if data:
                    signal = json.loads(data)
                    timestamp = signal.get('timestamp', 'N/A')
                    strategy = signal.get('strategy', 'N/A')
                    side = signal.get('side', 'N/A')
                    confidence = signal.get('confidence', 0)
                    print(f"  {symbol:6} | {side:5} | {strategy:6} | Conf: {confidence}% | Time: {timestamp}")

        # Signal fingerprints (dedup tracking)
        fingerprint_keys = await self.redis.keys('signals:fingerprint:*')
        print(f"\n🔍 Active Signal Fingerprints: {len(fingerprint_keys)} symbols")

        # Cooldowns
        cooldown_keys = await self.redis.keys('signals:cooldown:*')
        if cooldown_keys:
            print(f"\n⏱️ Active Cooldowns: {len(cooldown_keys)} symbols")
            for key in cooldown_keys[:5]:
                symbol = key.split(':')[-1]
                ttl = await self.redis.ttl(key)
                if ttl > 0:
                    print(f"  {symbol}: {ttl}s remaining")

    async def get_distribution_queues(self):
        """Display distribution queue status"""
        print("\n" + "="*80)
        print("DISTRIBUTION QUEUES")
        print("="*80)

        queues = {
            'Premium': 'distribution:premium:queue',
            'Basic': 'distribution:basic:queue',
            'Free': 'distribution:free:queue',
        }

        for tier, key in queues.items():
            queue_len = await self.redis.llen(key)
            print(f"\n💎 {tier} Queue: {queue_len} items")
            if queue_len > 0:
                items = await self.redis.lrange(key, 0, min(3, queue_len-1))
                for i, item in enumerate(items, 1):
                    print(f"  Item {i}: {item[:100]}..." if len(item) > 100 else f"  Item {i}: {item}")

    async def get_risk_metrics(self):
        """Display risk management metrics"""
        print("\n" + "="*80)
        print("RISK MANAGEMENT")
        print("="*80)

        risk_data = []

        # Simple value metrics
        metrics = [
            ('Daily P&L', 'risk:daily_pnl', 'currency'),
            ('Realized P&L', 'positions:pnl:realized:total', 'currency'),
            ('Unrealized P&L', 'positions:pnl:unrealized', 'currency'),
            ('Daily Trades', 'risk:daily_trades', None),
            ('Consecutive Losses', 'risk:consecutive_losses', None),
            ('Current Drawdown', 'risk:current_drawdown', 'json'),
            ('Max Drawdown', 'risk:drawdown:max', 'currency'),
            ('Halt Status', 'risk:halt:status', None),
            ('Emergency Active', 'risk:emergency:active', None),
            ('Current VaR', 'risk:var:current', 'currency'),
        ]

        for name, key, value_type in metrics:
            value = await self.redis.get(key)
            if value:
                try:
                    if value_type == 'currency':
                        value = f"${float(value):,.2f}"
                    elif value_type == 'json':
                        parsed = json.loads(value)
                        pct = parsed.get('drawdown_pct', 0.0)
                        current_val = parsed.get('current_value', 0.0)
                        hwm = parsed.get('high_water_mark', 0.0)
                        value = f"{pct:.2f}% (Acct ${current_val:,.2f} | HWM ${hwm:,.2f})"
                except Exception:
                    pass
                risk_data.append([name, value])
            else:
                risk_data.append([name, "N/A"])

        # Emergency timestamp
        emergency_ts = await self.redis.get('risk:emergency:timestamp')
        if emergency_ts:
            try:
                dt = datetime.fromisoformat(emergency_ts.replace('Z', '+00:00'))
                risk_data.append(['Emergency Timestamp', dt.strftime('%Y-%m-%d %H:%M:%S')])
            except:
                risk_data.append(['Emergency Timestamp', emergency_ts])

        if risk_data:
            print("\n" + tabulate(risk_data, headers=['Metric', 'Value'], tablefmt='simple'))

        # Correlation matrix
        correlation = await self.redis.get('risk:correlation:matrix')
        if correlation:
            print("\n📊 Correlation Matrix:")
            self._print_json(correlation)

        # Drawdown history (if available)
        history = await self.redis.lrange('risk:drawdown:history', 0, -1)
        if history:
            print("\n📉 Recent Drawdown Samples:")
            latest = []
            for entry in history[:5]:
                try:
                    item = json.loads(entry)
                except Exception:
                    continue
                latest.append([
                    item.get('timestamp', 'N/A'),
                    f"{float(item.get('drawdown_pct', 0)):0.2f}%",
                    f"${float(item.get('account_value', 0)):,.2f}",
                    f"${float(item.get('hwm', 0)):,.2f}",
                ])
            if latest:
                print(tabulate(latest, headers=['Time', 'Drawdown %', 'Account Value', 'HWM'], tablefmt='simple'))
        else:
            print("\n⚠️ Drawdown history missing (risk manager may not be running or recording).")

    async def get_positions(self):
        """Display open positions and exposure"""
        print("\n" + "="*80)
        print("POSITIONS")
        print("="*80)

        # Count
        count = await self.redis.get('positions:count')
        print(f"\n📦 Total Open Positions: {count or 0}")

        # Total exposure
        total_exposure = await self.redis.get('positions:exposure:total')
        if total_exposure:
            print(f"💰 Total Exposure: ${float(total_exposure):,.2f}")

        # Open positions
        position_keys = await self.redis.keys('positions:open:*')
        if position_keys:
            print(f"\n📋 Open Positions ({len(position_keys)} total):")

            position_table = []
            for key in position_keys[:10]:  # Show first 10
                pos_data = await self.redis.get(key)
                if pos_data:
                    try:
                        pos = json.loads(pos_data)
                        position_table.append([
                            pos.get('symbol', 'N/A'),
                            pos.get('side', 'N/A'),
                            pos.get('quantity', 0),
                            f"${pos.get('entry_price', 0):.2f}",
                            f"${pos.get('unrealized_pnl', 0):.2f}",
                            pos.get('strategy', 'N/A'),
                            pos.get('status', 'N/A'),
                        ])
                    except:
                        continue

            if position_table:
                headers = ['Symbol', 'Side', 'Qty', 'Entry', 'Unrealized P&L', 'Strategy', 'Status']
                print("\n" + tabulate(position_table, headers=headers, tablefmt='simple'))

            if len(position_keys) > 10:
                print(f"\n... and {len(position_keys) - 10} more positions")

        # Positions by symbol
        by_symbol_keys = await self.redis.keys('positions:by_symbol:*')
        if by_symbol_keys:
            print(f"\n🎯 Positions grouped by {len(by_symbol_keys)} symbols:")
            for key in by_symbol_keys:
                symbol = key.split(':')[-1]
                position_ids = await self.redis.smembers(key)
                if position_ids:
                    print(f"  {symbol}: {position_ids}")

        # Positions by strategy
        by_strategy_keys = await self.redis.keys('positions:by_strategy:*')
        if by_strategy_keys:
            print(f"\n📊 Positions grouped by strategies:")
            for key in by_strategy_keys:
                strategy = key.split(':')[-1]
                members = await self.redis.smembers(key)
                print(f"  {strategy}: {len(members)} positions")

        # Symbol exposures
        exposure_keys = await self.redis.keys('positions:exposure:*')
        if exposure_keys and len(exposure_keys) > 1:  # Skip if only total
            print("\n💼 Exposure by Symbol:")
            exposures = []
            for key in exposure_keys:
                if not key.endswith(':total'):
                    symbol = key.split(':')[-1]
                    exposure = await self.redis.get(key)
                    if exposure:
                        exposures.append([symbol, f"${float(exposure):,.2f}"])
            if exposures:
                print(tabulate(exposures[:10], headers=['Symbol', 'Exposure'], tablefmt='simple'))

        # Closed positions for today
        today = datetime.utcnow().strftime('%Y%m%d')
        closed_keys = await self.redis.keys(f'positions:closed:{today}:*')
        if closed_keys:
            print("\n✅ Closed Positions Today:")
            rows = []
            seen = set()
            for key in sorted(closed_keys):
                payload = await self.redis.get(key)
                if not payload:
                    continue
                try:
                    pos = json.loads(payload)
                except Exception:
                    continue

                realized = float(pos.get('realized_pnl', 0) or 0.0)
                commission = float(pos.get('commission', 0) or 0.0)
                strategy = (pos.get('strategy') or '').upper()

                # Skip reconciled placeholders (realized/commission zero and strategy flagged)
                if realized == 0.0 and commission == 0.0 and strategy == 'RECONCILED':
                    continue

                contract = pos.get('contract', {})
                dedupe_key = (pos.get('symbol'), contract.get('localSymbol'), pos.get('exit_time'))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                entry_time = pos.get('entry_time')
                exit_time = pos.get('exit_time')

                def to_et(ts: Optional[str]) -> str:
                    if not ts:
                        return 'N/A'
                    try:
                        ts_norm = ts.replace('Z', '+00:00')
                        dt = datetime.fromisoformat(ts_norm)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        dt_et = dt.astimezone(ZoneInfo('US/Eastern'))
                        return dt_et.strftime('%Y-%m-%d %H:%M:%S ET')
                    except Exception:
                        return ts

                rows.append([
                    pos.get('symbol', 'N/A'),
                    contract.get('localSymbol', contract.get('symbol', 'N/A')),
                    pos.get('side', 'N/A'),
                    f"${pos.get('entry_price', 0):.2f}",
                    to_et(entry_time),
                    f"${pos.get('exit_price', 0):.2f}",
                    to_et(exit_time),
                    f"${realized:.2f}",
                    f"${commission:.2f}",
                    (len(pos.get('realized_events', [])) or ''),
                ])

            if rows:
                headers = [
                    'Symbol',
                    'Contract',
                    'Side',
                    'Entry $',
                    'Entry (ET)',
                    'Exit $',
                    'Exit (ET)',
                    'Realized',
                    'Comm',
                    '#Fills',
                ]
                print(tabulate(rows[:10], headers=headers, tablefmt='simple'))
                if len(rows) > 10:
                    print(f"\n... and {len(rows) - 10} more closed positions")
                print(f"\nShowing {len(rows[:10])} of {len(rows)} reconciled closes.")
            else:
                print("  (No closed positions with realized P&L yet today)")

    async def get_orders(self):
        """Display order management data"""
        print("\n" + "="*80)
        print("ORDERS")
        print("="*80)

        order_categories = [
            ('Pending', 'orders:pending:*'),
            ('Active', 'orders:active:*'),
            ('Filled', 'orders:filled:*'),
            ('Rejected', 'orders:rejected:*'),
        ]

        for category, pattern in order_categories:
            keys = await self.redis.keys(pattern)
            if keys:
                print(f"\n🔄 {category} Orders: {len(keys)}")

                # Show first few orders
                for key in keys[:3]:
                    order_data = await self.redis.get(key)
                    if order_data:
                        try:
                            order = json.loads(order_data)
                            order_id = key.split(':')[-1]
                            symbol = order.get('symbol', 'N/A')
                            action = order.get('action', 'N/A')
                            qty = order.get('quantity', 0)
                            order_type = order.get('order_type', 'N/A')
                            print(f"  ID: {order_id[:8]}... | {symbol} | {action} {qty} | Type: {order_type}")
                        except:
                            print(f"  {key}: {order_data[:50]}...")

                if len(keys) > 3:
                    print(f"  ... and {len(keys) - 3} more")

        # Orders by symbol
        by_symbol_keys = await self.redis.keys('orders:by_symbol:*')
        if by_symbol_keys:
            print(f"\n📍 Orders tracked for {len(by_symbol_keys)} symbols")

    def _print_json(self, data: str, indent: int = 2):
        """Pretty print JSON data"""
        try:
            parsed = json.loads(data)
            print(json.dumps(parsed, indent=indent, default=str))
        except:
            print(data)

    async def display_all(self):
        """Display all Redis data"""
        if not await self.connect():
            return

        try:
            await self.get_portfolio_analytics()
            await self.get_signals()
            await self.get_distribution_queues()
            await self.get_risk_metrics()
            await self.get_positions()
            await self.get_orders()

            print("\n" + "="*80)
            print("✅ Data scan complete")
            print("="*80)

        except Exception as e:
            print(f"\n❌ Error during scan: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.redis.aclose()

async def main():
    """Main entry point"""
    viewer = RedisDataViewer()

    # Check for specific sections via command line args
    if len(sys.argv) > 1:
        section = sys.argv[1].lower()
        await viewer.connect()

        if section == 'portfolio':
            await viewer.get_portfolio_analytics()
        elif section == 'signals':
            await viewer.get_signals()
        elif section == 'distribution':
            await viewer.get_distribution_queues()
        elif section == 'risk':
            await viewer.get_risk_metrics()
        elif section == 'positions':
            await viewer.get_positions()
        elif section == 'orders':
            await viewer.get_orders()
        else:
            print(f"Unknown section: {section}")
            print("Available sections: portfolio, signals, distribution, risk, positions, orders")

        await viewer.redis.aclose()
    else:
        # Display everything
        await viewer.display_all()

if __name__ == "__main__":
    print("🔍 Redis Data Viewer for QuantiCity Capital")
    print("-" * 40)
    asyncio.run(main())
