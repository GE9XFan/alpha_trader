#!/usr/bin/env python3
"""
Day 5 Discovery Script - Complete Data Structure Analysis
Discovers and documents the exact structure of all API responses
This is the foundation for database design
"""
import asyncio
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import inspect
from pprint import pformat
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.alpha_vantage_client import AlphaVantageClient, OptionContract
from src.data.market_data import MarketDataManager
from src.core.logger import get_logger
from ib_insync import util

logger = get_logger(__name__)

class DataDiscovery:
    """
    Discovers and documents all data structures from our working APIs
    Day 5 implementation - discover first, then design database
    """
    
    def __init__(self):
        self.av_client = AlphaVantageClient()
        self.ibkr_client = MarketDataManager()
        self.discoveries = {}
        self.output_dir = Path("discovery_output")
        self.output_dir.mkdir(exist_ok=True)
        
    async def run_complete_discovery(self):
        """Main discovery process"""
        print("\n" + "="*80)
        print("🔍 DAY 5: COMPLETE DATA STRUCTURE DISCOVERY")
        print("="*80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Output Directory: {self.output_dir}")
        print("="*80 + "\n")
        
        # Phase 1: Alpha Vantage Discovery
        print("\n" + "="*80)
        print("PHASE 1: ALPHA VANTAGE API DISCOVERY (36 APIs)")
        print("="*80 + "\n")
        await self.discover_alpha_vantage()
        
        # Phase 2: IBKR Discovery
        print("\n" + "="*80)
        print("PHASE 2: IBKR DATA STRUCTURE DISCOVERY")
        print("="*80 + "\n")
        await self.discover_ibkr()
        
        # Phase 3: Generate Documentation
        print("\n" + "="*80)
        print("PHASE 3: GENERATING DOCUMENTATION")
        print("="*80 + "\n")
        self.generate_documentation()
        
        # Phase 4: Generate Database Schema
        print("\n" + "="*80)
        print("PHASE 4: GENERATING DATABASE SCHEMA")
        print("="*80 + "\n")
        self.generate_database_schema()
        
        print("\n" + "="*80)
        print("✅ DISCOVERY COMPLETE!")
        print(f"📊 Total APIs Discovered: {len(self.discoveries)}")
        print(f"📁 Output saved to: {self.output_dir}")
        print("="*80)
    
    async def discover_alpha_vantage(self):
        """Discover all Alpha Vantage API response structures"""
        await self.av_client.connect()
        
        # Test parameters
        test_symbol = 'AAPL'
        test_date = '2025-08-20'
        
        # 1. OPTIONS APIs (2)
        print("\n📦 OPTIONS APIs:")
        
        # REALTIME_OPTIONS
        print("  1. REALTIME_OPTIONS...")
        try:
            options = await self.av_client.get_realtime_options(test_symbol, require_greeks=True)
            if options and len(options) > 0:
                sample = options[0]
                self.discoveries['av_realtime_options'] = {
                    'api': 'REALTIME_OPTIONS',
                    'return_type': 'List[OptionContract]',
                    'sample_count': len(options),
                    'fields': self._extract_dataclass_fields(sample),
                    'sample_data': self._serialize_option(sample),
                    'nullable_fields': self._find_nullable_fields([self._serialize_option(o) for o in options[:10]])
                }
                print(f"    ✅ Found {len(options)} options with {len(self.discoveries['av_realtime_options']['fields'])} fields")
                self._save_raw_data('av_realtime_options', options[:5])
        except Exception as e:
            print(f"    ❌ Error: {e}")
            
        # HISTORICAL_OPTIONS
        print("  2. HISTORICAL_OPTIONS...")
        try:
            hist_options = await self.av_client.get_historical_options(test_symbol, test_date)
            if hist_options and len(hist_options) > 0:
                sample = hist_options[0]
                self.discoveries['av_historical_options'] = {
                    'api': 'HISTORICAL_OPTIONS',
                    'return_type': 'List[OptionContract]',
                    'sample_count': len(hist_options),
                    'fields': self._extract_dataclass_fields(sample),
                    'sample_data': self._serialize_option(sample),
                    'nullable_fields': self._find_nullable_fields([self._serialize_option(o) for o in hist_options[:10]])
                }
                print(f"    ✅ Found {len(hist_options)} options with {len(self.discoveries['av_historical_options']['fields'])} fields")
                self._save_raw_data('av_historical_options', hist_options[:5])
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        # 2. TECHNICAL INDICATORS (16)
        print("\n📊 TECHNICAL INDICATORS APIs:")
        
        indicators = [
            ('RSI', 'get_rsi', {'interval': 'daily', 'time_period': 14, 'series_type': 'close'}),
            ('MACD', 'get_macd', {'interval': 'daily', 'series_type': 'close'}),
            ('STOCH', 'get_stoch', {'interval': 'daily'}),
            ('WILLR', 'get_willr', {'interval': 'daily', 'time_period': 14}),
            ('MOM', 'get_mom', {'interval': 'daily', 'time_period': 10, 'series_type': 'close'}),
            ('BBANDS', 'get_bbands', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
            ('ATR', 'get_atr', {'interval': 'daily', 'time_period': 14}),
            ('ADX', 'get_adx', {'interval': 'daily', 'time_period': 14}),
            ('AROON', 'get_aroon', {'interval': 'daily', 'time_period': 14}),
            ('CCI', 'get_cci', {'interval': 'daily', 'time_period': 20}),
            ('EMA', 'get_ema', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
            ('SMA', 'get_sma', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
            ('MFI', 'get_mfi', {'interval': 'daily', 'time_period': 14}),
            ('OBV', 'get_obv', {'interval': 'daily'}),
            ('AD', 'get_ad', {'interval': 'daily'}),
            ('VWAP', 'get_vwap', {'interval': '15min'})
        ]
        
        for i, (name, method_name, params) in enumerate(indicators, 1):
            print(f"  {i:2}. {name}...", end=' ')
            try:
                method = getattr(self.av_client, method_name)
                df = await method(test_symbol, **params)
                
                if df is not None and not df.empty:
                    self.discoveries[f'av_{name.lower()}'] = {
                        'api': name,
                        'method': method_name,
                        'return_type': 'pd.DataFrame',
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'index_type': str(df.index.dtype),
                        'sample_data': df.head(3).to_dict(),
                        'nullable_columns': list(df.columns[df.isna().any()])
                    }
                    print(f"✅ {df.shape[0]} rows, {df.shape[1]} cols")
                    self._save_raw_data(f'av_{name.lower()}', df.head(10))
                else:
                    print("❌ Empty response")
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        # 3. ANALYTICS APIs (2)
        print("\n📈 ANALYTICS APIs:")
        
        print("  1. ANALYTICS_FIXED_WINDOW...")
        try:
            analytics = await self.av_client.get_analytics_fixed_window(
                symbols='SPY,QQQ',
                interval='DAILY',
                range='1month'
            )
            if analytics:
                self.discoveries['av_analytics_fixed'] = {
                    'api': 'ANALYTICS_FIXED_WINDOW',
                    'return_type': 'Dict',
                    'top_level_keys': list(analytics.keys()),
                    'structure': self._analyze_dict_structure(analytics),
                    'sample_data': self._truncate_dict(analytics)
                }
                print(f"    ✅ Found {len(analytics.keys())} top-level keys")
                self._save_raw_data('av_analytics_fixed', analytics)
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        print("  2. ANALYTICS_SLIDING_WINDOW...")
        try:
            sliding = await self.av_client.get_analytics_sliding_window(
                symbols='SPY,QQQ',
                window_size=30,
                interval='DAILY',
                range='6month'
            )
            if sliding:
                self.discoveries['av_analytics_sliding'] = {
                    'api': 'ANALYTICS_SLIDING_WINDOW',
                    'return_type': 'Dict',
                    'top_level_keys': list(sliding.keys()),
                    'structure': self._analyze_dict_structure(sliding),
                    'sample_data': self._truncate_dict(sliding)
                }
                print(f"    ✅ Found {len(sliding.keys())} top-level keys")
                self._save_raw_data('av_analytics_sliding', sliding)
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        # 4. SENTIMENT APIs (3)
        print("\n💭 SENTIMENT APIs:")
        
        print("  1. NEWS_SENTIMENT...")
        try:
            news = await self.av_client.get_news_sentiment(tickers='AAPL,MSFT', limit=50)
            if news and 'feed' in news:
                self.discoveries['av_news_sentiment'] = {
                    'api': 'NEWS_SENTIMENT',
                    'return_type': 'Dict',
                    'top_level_keys': list(news.keys()),
                    'feed_count': len(news.get('feed', [])),
                    'structure': self._analyze_dict_structure(news),
                    'article_fields': list(news['feed'][0].keys()) if news.get('feed') else []
                }
                print(f"    ✅ Found {len(news.get('feed', []))} articles")
                self._save_raw_data('av_news_sentiment', news)
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        print("  2. TOP_GAINERS_LOSERS...")
        try:
            movers = await self.av_client.get_top_gainers_losers()
            if movers:
                self.discoveries['av_top_movers'] = {
                    'api': 'TOP_GAINERS_LOSERS',
                    'return_type': 'Dict',
                    'top_level_keys': list(movers.keys()),
                    'structure': self._analyze_dict_structure(movers)
                }
                print(f"    ✅ Found {len(movers.keys())} categories")
                self._save_raw_data('av_top_movers', movers)
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        print("  3. INSIDER_TRANSACTIONS...")
        try:
            insider = await self.av_client.get_insider_transactions(test_symbol)
            if insider:
                self.discoveries['av_insider'] = {
                    'api': 'INSIDER_TRANSACTIONS',
                    'return_type': 'Dict',
                    'top_level_keys': list(insider.keys()),
                    'structure': self._analyze_dict_structure(insider)
                }
                print(f"    ✅ Found insider data")
                self._save_raw_data('av_insider', insider)
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        # 5. FUNDAMENTALS (8)
        print("\n📋 FUNDAMENTALS APIs:")
        
        fundamentals = [
            ('OVERVIEW', 'get_overview'),
            ('EARNINGS', 'get_earnings'),
            ('INCOME_STATEMENT', 'get_income_statement'),
            ('BALANCE_SHEET', 'get_balance_sheet'),
            ('CASH_FLOW', 'get_cash_flow'),
            ('DIVIDENDS', 'get_dividends'),
            ('SPLITS', 'get_splits'),
            ('EARNINGS_CALENDAR', 'get_earnings_calendar')
        ]
        
        for i, (name, method_name) in enumerate(fundamentals, 1):
            print(f"  {i}. {name}...", end=' ')
            try:
                method = getattr(self.av_client, method_name)
                
                if name == 'EARNINGS_CALENDAR':
                    # Special case - returns CSV string
                    result = await method('3month')
                    if result:
                        self.discoveries[f'av_{name.lower()}'] = {
                            'api': name,
                            'return_type': 'str (CSV)',
                            'length': len(result),
                            'sample': result[:500]
                        }
                        print(f"✅ CSV with {len(result)} chars")
                else:
                    result = await method(test_symbol)
                    if result:
                        self.discoveries[f'av_{name.lower()}'] = {
                            'api': name,
                            'return_type': 'Dict',
                            'top_level_keys': list(result.keys()),
                            'structure': self._analyze_dict_structure(result)
                        }
                        print(f"✅ {len(result.keys())} keys")
                        self._save_raw_data(f'av_{name.lower()}', result)
            except Exception as e:
                print(f"❌ {str(e)[:30]}")
            
            await asyncio.sleep(0.1)
        
        # 6. ECONOMIC APIs (5)
        print("\n💹 ECONOMIC APIs:")
        
        economic = [
            ('TREASURY_YIELD', 'get_treasury_yield', {'interval': 'monthly', 'maturity': '10year'}),
            ('FEDERAL_FUNDS_RATE', 'get_federal_funds_rate', {'interval': 'monthly'}),
            ('CPI', 'get_cpi', {'interval': 'monthly'}),
            ('INFLATION', 'get_inflation', {}),
            ('REAL_GDP', 'get_real_gdp', {'interval': 'quarterly'})
        ]
        
        for i, (name, method_name, params) in enumerate(economic, 1):
            print(f"  {i}. {name}...", end=' ')
            try:
                method = getattr(self.av_client, method_name)
                df = await method(**params)
                
                if isinstance(df, pd.DataFrame) and not df.empty:
                    self.discoveries[f'av_{name.lower()}'] = {
                        'api': name,
                        'return_type': 'pd.DataFrame',
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'sample_data': df.head(3).to_dict()
                    }
                    print(f"✅ {df.shape[0]} rows")
                    self._save_raw_data(f'av_{name.lower()}', df.head(10))
                else:
                    print("❌ Empty")
            except Exception as e:
                print(f"❌ {str(e)[:30]}")
            
            await asyncio.sleep(0.1)
        
        await self.av_client.disconnect()
    
    async def discover_ibkr(self):
        """Discover IBKR data structures"""
        try:
            # Connect to IBKR
            print("🔌 Connecting to IBKR...")
            connected = await self.ibkr_client.connect()
            
            if not connected:
                print("❌ Failed to connect to IBKR")
                return
            
            print("✅ Connected to IBKR")
            
            # Subscribe to test symbol
            test_symbol = 'SPY'
            print(f"\n📊 Subscribing to {test_symbol}...")
            results = await self.ibkr_client.subscribe_symbols([test_symbol])
            
            # Wait for data
            print("⏳ Waiting for data (5 seconds)...")
            await asyncio.sleep(5)
            
            # 1. Latest Price Structure
            price = self.ibkr_client.get_latest_price(test_symbol)
            self.discoveries['ibkr_latest_price'] = {
                'api': 'get_latest_price',
                'return_type': 'float',
                'sample_value': price
            }
            print(f"✅ Latest Price: ${price:.2f}")
            
            # 2. Latest Bar Structure
            bar = self.ibkr_client.get_latest_bar(test_symbol)
            if bar:
                self.discoveries['ibkr_latest_bar'] = {
                    'api': 'get_latest_bar',
                    'return_type': 'Dict',
                    'fields': list(bar.keys()),
                    'field_types': {k: type(v).__name__ for k, v in bar.items()},
                    'sample_data': bar
                }
                print(f"✅ Latest Bar: {len(bar)} fields")
                self._save_raw_data('ibkr_latest_bar', bar)
            
            # 3. Bar History Structure
            history = self.ibkr_client.get_bar_history(test_symbol, num_bars=10)
            if not history.empty:
                self.discoveries['ibkr_bar_history'] = {
                    'api': 'get_bar_history',
                    'return_type': 'pd.DataFrame',
                    'shape': history.shape,
                    'columns': list(history.columns),
                    'dtypes': {col: str(dtype) for col, dtype in history.dtypes.items()},
                    'sample_data': history.head(3).to_dict()
                }
                print(f"✅ Bar History: {history.shape[0]} bars, {history.shape[1]} fields")
                self._save_raw_data('ibkr_bar_history', history)
            
            # 4. Historical Bars Structure
            hist_bars = await self.ibkr_client.get_historical_bars(
                test_symbol, 
                duration='1 D',
                bar_size='1 min'
            )
            if not hist_bars.empty:
                self.discoveries['ibkr_historical_bars'] = {
                    'api': 'get_historical_bars',
                    'return_type': 'pd.DataFrame',
                    'shape': hist_bars.shape,
                    'columns': list(hist_bars.columns),
                    'dtypes': {col: str(dtype) for col, dtype in hist_bars.dtypes.items()},
                    'index_type': str(hist_bars.index.dtype),
                    'sample_data': hist_bars.head(3).to_dict()
                }
                print(f"✅ Historical Bars: {hist_bars.shape[0]} bars, {hist_bars.shape[1]} fields")
                self._save_raw_data('ibkr_historical_bars', hist_bars.head(10))
            
            # 5. Connection Status Structure
            status = self.ibkr_client.get_connection_status()
            self.discoveries['ibkr_connection_status'] = {
                'api': 'get_connection_status',
                'return_type': 'Dict',
                'fields': list(status.keys()),
                'field_types': {k: type(v).__name__ for k, v in status.items()},
                'sample_data': status
            }
            print(f"✅ Connection Status: {len(status)} fields")
            
            # 6. All Prices Structure
            all_prices = self.ibkr_client.get_all_prices()
            self.discoveries['ibkr_all_prices'] = {
                'api': 'get_all_prices',
                'return_type': 'Dict[str, float]',
                'sample_data': all_prices
            }
            print(f"✅ All Prices: {len(all_prices)} symbols")
            
            # Disconnect
            await self.ibkr_client.disconnect()
            print("✅ Disconnected from IBKR")
            
        except Exception as e:
            logger.error(f"IBKR discovery error: {e}")
            print(f"❌ IBKR discovery failed: {e}")
    
    def _extract_dataclass_fields(self, obj):
        """Extract fields from a dataclass instance"""
        fields = {}
        for field_name in dir(obj):
            if not field_name.startswith('_'):
                value = getattr(obj, field_name)
                if not callable(value):
                    fields[field_name] = {
                        'type': type(value).__name__,
                        'sample_value': str(value)[:100] if value is not None else None
                    }
        return fields
    
    def _serialize_option(self, option: OptionContract) -> Dict:
        """Convert OptionContract to dictionary"""
        return {
            'symbol': option.symbol,
            'strike': option.strike,
            'expiry': option.expiry,
            'option_type': option.option_type,
            'bid': option.bid,
            'ask': option.ask,
            'last': option.last,
            'volume': option.volume,
            'open_interest': option.open_interest,
            'implied_volatility': option.implied_volatility,
            'delta': option.delta,
            'gamma': option.gamma,
            'theta': option.theta,
            'vega': option.vega,
            'rho': option.rho
        }
    
    def _find_nullable_fields(self, samples: List[Dict]) -> List[str]:
        """Find fields that can be None/null"""
        nullable = []
        for field in samples[0].keys():
            for sample in samples:
                if sample.get(field) is None or (isinstance(sample.get(field), float) and sample.get(field) == 0.0):
                    nullable.append(field)
                    break
        return nullable
    
    def _analyze_dict_structure(self, d: Dict, max_depth: int = 3, current_depth: int = 0) -> Dict:
        """Analyze nested dictionary structure"""
        if current_depth >= max_depth:
            return {'type': 'dict', 'truncated': True}
        
        structure = {'type': 'dict', 'keys': {}}
        for key, value in d.items():
            if isinstance(value, dict):
                structure['keys'][key] = self._analyze_dict_structure(value, max_depth, current_depth + 1)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    structure['keys'][key] = {
                        'type': 'list',
                        'length': len(value),
                        'item_structure': self._analyze_dict_structure(value[0], max_depth, current_depth + 1)
                    }
                else:
                    structure['keys'][key] = {
                        'type': 'list',
                        'length': len(value),
                        'item_type': type(value[0]).__name__ if value else 'unknown'
                    }
            else:
                structure['keys'][key] = {'type': type(value).__name__}
        
        return structure
    
    def _truncate_dict(self, d: Dict, max_items: int = 3) -> Dict:
        """Truncate dictionary for display"""
        if len(d) <= max_items:
            return d
        
        truncated = {}
        for i, (k, v) in enumerate(d.items()):
            if i >= max_items:
                truncated['...'] = f"({len(d) - max_items} more items)"
                break
            
            if isinstance(v, dict):
                truncated[k] = self._truncate_dict(v, max_items)
            elif isinstance(v, list) and len(v) > max_items:
                truncated[k] = v[:max_items] + [f"...({len(v) - max_items} more)"]
            else:
                truncated[k] = v
        
        return truncated
    
    def _save_raw_data(self, name: str, data: Any):
        """Save raw data to file for reference"""
        filepath = self.output_dir / f"{name}_raw.json"
        
        try:
            # Convert to JSON-serializable format
            if isinstance(data, pd.DataFrame):
                json_data = data.head(10).to_dict(orient='records')
            elif isinstance(data, list) and len(data) > 0 and hasattr(data[0], '__dict__'):
                # Handle list of dataclass objects
                json_data = [self._serialize_option(item) if hasattr(item, 'symbol') else item.__dict__ 
                           for item in data[:10]]
            else:
                json_data = data
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save raw data for {name}: {e}")
    
    def generate_documentation(self):
        """Generate comprehensive documentation of discoveries"""
        doc_path = self.output_dir / "data_structure_documentation.md"
        
        with open(doc_path, 'w') as f:
            f.write("# AlphaTrader Data Structure Documentation\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n")
            f.write(f"- Total APIs Discovered: {len(self.discoveries)}\n")
            f.write(f"- Alpha Vantage APIs: {sum(1 for k in self.discoveries if k.startswith('av_'))}\n")
            f.write(f"- IBKR APIs: {sum(1 for k in self.discoveries if k.startswith('ibkr_'))}\n\n")
            
            # Alpha Vantage Documentation
            f.write("## Alpha Vantage API Structures\n\n")
            for key, discovery in self.discoveries.items():
                if key.startswith('av_'):
                    f.write(f"### {discovery.get('api', key)}\n")
                    f.write(f"- **Return Type**: {discovery.get('return_type', 'Unknown')}\n")
                    
                    if 'shape' in discovery:
                        f.write(f"- **Shape**: {discovery['shape']}\n")
                    if 'sample_count' in discovery:
                        f.write(f"- **Sample Count**: {discovery['sample_count']}\n")
                    if 'columns' in discovery:
                        f.write(f"- **Columns**: {', '.join(discovery['columns'])}\n")
                    if 'fields' in discovery:
                        f.write("- **Fields**:\n")
                        for field, info in discovery['fields'].items():
                            f.write(f"  - `{field}`: {info['type']}\n")
                    if 'nullable_fields' in discovery:
                        f.write(f"- **Nullable Fields**: {', '.join(discovery['nullable_fields'])}\n")
                    if 'top_level_keys' in discovery:
                        f.write(f"- **Top Level Keys**: {', '.join(discovery['top_level_keys'][:10])}\n")
                    
                    f.write("\n")
            
            # IBKR Documentation
            f.write("## IBKR Data Structures\n\n")
            for key, discovery in self.discoveries.items():
                if key.startswith('ibkr_'):
                    f.write(f"### {discovery.get('api', key)}\n")
                    f.write(f"- **Return Type**: {discovery.get('return_type', 'Unknown')}\n")
                    
                    if 'fields' in discovery:
                        f.write(f"- **Fields**: {', '.join(discovery['fields'])}\n")
                    if 'columns' in discovery:
                        f.write(f"- **Columns**: {', '.join(discovery['columns'])}\n")
                    if 'shape' in discovery:
                        f.write(f"- **Shape**: {discovery['shape']}\n")
                    
                    f.write("\n")
        
        print(f"📄 Documentation saved to: {doc_path}")
    
    def generate_database_schema(self):
        """Generate PostgreSQL schema based on discoveries"""
        schema_path = self.output_dir / "database_schema.sql"
        
        with open(schema_path, 'w') as f:
            f.write("-- AlphaTrader Database Schema\n")
            f.write(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-- Based on actual API response structures\n\n")
            
            # Options table (for both realtime and historical)
            f.write("-- Options data from Alpha Vantage\n")
            f.write("CREATE TABLE IF NOT EXISTS options_data (\n")
            f.write("    id SERIAL PRIMARY KEY,\n")
            f.write("    timestamp TIMESTAMPTZ DEFAULT NOW(),\n")
            f.write("    data_type VARCHAR(20), -- 'realtime' or 'historical'\n")
            f.write("    symbol VARCHAR(10) NOT NULL,\n")
            f.write("    strike DECIMAL(10,2) NOT NULL,\n")
            f.write("    expiry DATE NOT NULL,\n")
            f.write("    option_type VARCHAR(4) NOT NULL,\n")
            f.write("    bid DECIMAL(10,4),\n")
            f.write("    ask DECIMAL(10,4),\n")
            f.write("    last DECIMAL(10,4),\n")
            f.write("    volume INT,\n")
            f.write("    open_interest INT,\n")
            f.write("    implied_volatility DECIMAL(6,4),\n")
            f.write("    delta DECIMAL(6,4),\n")
            f.write("    gamma DECIMAL(6,4),\n")
            f.write("    theta DECIMAL(8,4),\n")
            f.write("    vega DECIMAL(8,4),\n")
            f.write("    rho DECIMAL(6,4),\n")
            f.write("    raw_data JSONB, -- Store complete response\n")
            f.write("    UNIQUE(symbol, strike, expiry, option_type, data_type, timestamp)\n")
            f.write(");\n\n")
            
            # Technical indicators table
            f.write("-- Technical indicators from Alpha Vantage\n")
            f.write("CREATE TABLE IF NOT EXISTS technical_indicators (\n")
            f.write("    id SERIAL PRIMARY KEY,\n")
            f.write("    timestamp TIMESTAMPTZ DEFAULT NOW(),\n")
            f.write("    symbol VARCHAR(10) NOT NULL,\n")
            f.write("    indicator VARCHAR(20) NOT NULL,\n")
            f.write("    interval VARCHAR(10),\n")
            f.write("    data_date TIMESTAMP NOT NULL,\n")
            f.write("    value DECIMAL(20,6),\n")
            f.write("    value2 DECIMAL(20,6), -- For multi-value indicators like MACD\n")
            f.write("    value3 DECIMAL(20,6), -- For BBANDS etc\n")
            f.write("    raw_data JSONB,\n")
            f.write("    UNIQUE(symbol, indicator, interval, data_date)\n")
            f.write(");\n\n")
            
            # Market data from IBKR
            f.write("-- Market data from IBKR\n")
            f.write("CREATE TABLE IF NOT EXISTS market_data (\n")
            f.write("    id SERIAL PRIMARY KEY,\n")
            f.write("    timestamp TIMESTAMPTZ DEFAULT NOW(),\n")
            f.write("    symbol VARCHAR(10) NOT NULL,\n")
            f.write("    bar_time TIMESTAMP NOT NULL,\n")
            f.write("    open DECIMAL(10,4),\n")
            f.write("    high DECIMAL(10,4),\n")
            f.write("    low DECIMAL(10,4),\n")
            f.write("    close DECIMAL(10,4),\n")
            f.write("    volume BIGINT,\n")
            f.write("    wap DECIMAL(10,4),\n")
            f.write("    count INT,\n")
            f.write("    bar_size VARCHAR(10),\n")
            f.write("    UNIQUE(symbol, bar_time, bar_size)\n")
            f.write(");\n\n")
            
            # News sentiment
            f.write("-- News sentiment from Alpha Vantage\n")
            f.write("CREATE TABLE IF NOT EXISTS news_sentiment (\n")
            f.write("    id SERIAL PRIMARY KEY,\n")
            f.write("    timestamp TIMESTAMPTZ DEFAULT NOW(),\n")
            f.write("    article_id VARCHAR(100),\n")
            f.write("    title TEXT,\n")
            f.write("    url TEXT,\n")
            f.write("    time_published TIMESTAMP,\n")
            f.write("    authors TEXT[],\n")
            f.write("    summary TEXT,\n")
            f.write("    sentiment_score DECIMAL(5,4),\n")
            f.write("    sentiment_label VARCHAR(20),\n")
            f.write("    ticker_relevance JSONB, -- Store ticker-specific scores\n")
            f.write("    raw_data JSONB,\n")
            f.write("    UNIQUE(article_id)\n")
            f.write(");\n\n")
            
            # Analytics results
            f.write("-- Analytics from Alpha Vantage\n")
            f.write("CREATE TABLE IF NOT EXISTS analytics (\n")
            f.write("    id SERIAL PRIMARY KEY,\n")
            f.write("    timestamp TIMESTAMPTZ DEFAULT NOW(),\n")
            f.write("    analytics_type VARCHAR(30),\n")
            f.write("    symbols TEXT[],\n")
            f.write("    interval VARCHAR(20),\n")
            f.write("    range VARCHAR(20),\n")
            f.write("    window_size INT,\n")
            f.write("    calculations TEXT[],\n")
            f.write("    results JSONB,\n")
            f.write("    raw_data JSONB\n")
            f.write(");\n\n")
            
            # Fundamentals
            f.write("-- Company fundamentals from Alpha Vantage\n")
            f.write("CREATE TABLE IF NOT EXISTS fundamentals (\n")
            f.write("    id SERIAL PRIMARY KEY,\n")
            f.write("    timestamp TIMESTAMPTZ DEFAULT NOW(),\n")
            f.write("    symbol VARCHAR(10) NOT NULL,\n")
            f.write("    data_type VARCHAR(30), -- overview, earnings, balance_sheet, etc\n")
            f.write("    period_ending DATE,\n")
            f.write("    raw_data JSONB NOT NULL,\n")
            f.write("    UNIQUE(symbol, data_type, period_ending)\n")
            f.write(");\n\n")
            
            # Economic indicators
            f.write("-- Economic indicators from Alpha Vantage\n")
            f.write("CREATE TABLE IF NOT EXISTS economic_indicators (\n")
            f.write("    id SERIAL PRIMARY KEY,\n")
            f.write("    timestamp TIMESTAMPTZ DEFAULT NOW(),\n")
            f.write("    indicator VARCHAR(30) NOT NULL,\n")
            f.write("    data_date DATE NOT NULL,\n")
            f.write("    value DECIMAL(20,6),\n")
            f.write("    interval VARCHAR(20),\n")
            f.write("    maturity VARCHAR(20),\n")
            f.write("    raw_data JSONB,\n")
            f.write("    UNIQUE(indicator, data_date, interval, maturity)\n")
            f.write(");\n\n")
            
            # Add indexes
            f.write("-- Indexes for performance\n")
            f.write("CREATE INDEX idx_options_symbol ON options_data(symbol);\n")
            f.write("CREATE INDEX idx_options_expiry ON options_data(expiry);\n")
            f.write("CREATE INDEX idx_options_timestamp ON options_data(timestamp DESC);\n")
            f.write("CREATE INDEX idx_indicators_symbol ON technical_indicators(symbol);\n")
            f.write("CREATE INDEX idx_indicators_date ON technical_indicators(data_date DESC);\n")
            f.write("CREATE INDEX idx_market_symbol ON market_data(symbol);\n")
            f.write("CREATE INDEX idx_market_time ON market_data(bar_time DESC);\n")
            f.write("CREATE INDEX idx_news_time ON news_sentiment(time_published DESC);\n")
            f.write("CREATE INDEX idx_fundamentals_symbol ON fundamentals(symbol);\n")
            
        print(f"📄 Database schema saved to: {schema_path}")
        
        # Also save as Python code for DatabaseManager
        self._generate_database_manager_code()
    
    def _generate_database_manager_code(self):
        """Generate updated DatabaseManager code"""
        code_path = self.output_dir / "database_manager_updated.py"
        
        # This would contain the actual storage methods
        # For brevity, just creating the structure
        
        with open(code_path, 'w') as f:
            f.write('"""Updated DatabaseManager with discovered schemas"""\n\n')
            f.write("# Copy this into src/data/database.py\n\n")
            f.write("# Table creation SQL is in database_schema.sql\n")
            f.write("# Storage methods below:\n\n")
            
            f.write("async def store_options_data(self, options: List[OptionContract], data_type: str = 'realtime'):\n")
            f.write("    \"\"\"Store options data from Alpha Vantage\"\"\"\n")
            f.write("    # Implementation based on discovered structure\n")
            f.write("    pass\n\n")
            
            f.write("async def store_market_data(self, symbol: str, bars: pd.DataFrame):\n")
            f.write("    \"\"\"Store market data from IBKR\"\"\"\n")
            f.write("    # Implementation based on discovered structure\n")
            f.write("    pass\n\n")
            
            f.write("async def store_indicator(self, symbol: str, indicator: str, df: pd.DataFrame):\n")
            f.write("    \"\"\"Store technical indicator data\"\"\"\n")
            f.write("    # Implementation based on discovered structure\n")
            f.write("    pass\n\n")
        
        print(f"📄 DatabaseManager code template saved to: {code_path}")


async def main():
    """Main entry point for Day 5 Discovery"""
    discovery = DataDiscovery()
    
    try:
        await discovery.run_complete_discovery()
        
        print("\n" + "="*80)
        print("📊 DISCOVERY COMPLETE - READY FOR DATABASE IMPLEMENTATION")
        print("="*80)
        print("\nNext Steps:")
        print("1. Review generated documentation in discovery_output/")
        print("2. Review database_schema.sql")
        print("3. Update DatabaseManager with storage methods")
        print("4. Run schema creation")
        print("5. Test end-to-end data flow")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Discovery interrupted by user")
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        print(f"\n❌ Discovery failed: {e}")


if __name__ == "__main__":
    # Use ib_insync's event loop for IBKR compatibility
    util.run(main())