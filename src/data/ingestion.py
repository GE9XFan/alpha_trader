"""Data ingestion module - Phase 4.1 with caching"""

import sys
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.foundation.config_manager import ConfigManager
from src.data.cache_manager import get_cache  # ADD THIS


class DataIngestion:
    def __init__(self):
        self.config = ConfigManager()
        self.engine = create_engine(self.config.database_url)
        self.cache = get_cache()  # ADD THIS
    
    def ingest_options_data(self, api_response, symbol):
        """
        Ingest REALTIME_OPTIONS data into database
        Phase 4.1: Cache after successful ingestion
        """
        if not api_response or 'data' not in api_response:
            print("No data to ingest")
            return 0
        
        options_data = api_response['data']
        print(f"Processing {len(options_data)} option contracts...")
        
        records_inserted = 0
        records_updated = 0
        
        with self.engine.connect() as conn:
            for contract in options_data:
                try:
                    # Parse and clean data
                    record = {
                        'contract_id': contract['contractID'],
                        'symbol': contract['symbol'],
                        'expiration': contract['expiration'],
                        'strike': self._to_decimal(contract['strike']),
                        'option_type': contract['type'],
                        'last_price': self._to_decimal(contract.get('last')),
                        'mark': self._to_decimal(contract.get('mark')),
                        'bid': self._to_decimal(contract.get('bid')),
                        'bid_size': self._to_int(contract.get('bid_size')),
                        'ask': self._to_decimal(contract.get('ask')),
                        'ask_size': self._to_int(contract.get('ask_size')),
                        'volume': self._to_int(contract.get('volume')),
                        'open_interest': self._to_int(contract.get('open_interest')),
                        'date': contract.get('date'),
                        'implied_volatility': self._to_decimal(contract.get('implied_volatility')),
                        'delta': self._to_decimal(contract.get('delta')),
                        'gamma': self._to_decimal(contract.get('gamma')),
                        'theta': self._to_decimal(contract.get('theta')),
                        'vega': self._to_decimal(contract.get('vega')),
                        'rho': self._to_decimal(contract.get('rho')),
                        'updated_at': datetime.now()
                    }
                    
                    # Check if contract exists
                    existing = conn.execute(
                        text("SELECT id FROM av_realtime_options WHERE contract_id = :contract_id"),
                        {'contract_id': record['contract_id']}
                    ).fetchone()
                    
                    if existing:
                        # Update existing record
                        update_query = text("""
                            UPDATE av_realtime_options 
                            SET last_price = :last_price,
                                mark = :mark,
                                bid = :bid,
                                bid_size = :bid_size,
                                ask = :ask,
                                ask_size = :ask_size,
                                volume = :volume,
                                open_interest = :open_interest,
                                implied_volatility = :implied_volatility,
                                delta = :delta,
                                gamma = :gamma,
                                theta = :theta,
                                vega = :vega,
                                rho = :rho,
                                updated_at = :updated_at
                            WHERE contract_id = :contract_id
                        """)
                        conn.execute(update_query, record)
                        records_updated += 1
                    else:
                        # Insert new record
                        insert_query = text("""
                            INSERT INTO av_realtime_options 
                            (contract_id, symbol, expiration, strike, option_type,
                             last_price, mark, bid, bid_size, ask, ask_size,
                             volume, open_interest, date, implied_volatility,
                             delta, gamma, theta, vega, rho, updated_at)
                            VALUES 
                            (:contract_id, :symbol, :expiration, :strike, :option_type,
                             :last_price, :mark, :bid, :bid_size, :ask, :ask_size,
                             :volume, :open_interest, :date, :implied_volatility,
                             :delta, :gamma, :theta, :vega, :rho, :updated_at)
                        """)
                        conn.execute(insert_query, record)
                        records_inserted += 1
                        
                except Exception as e:
                    print(f"Error processing contract {contract.get('contractID', 'unknown')}: {e}")
                    continue
            
            conn.commit()
        
        # PHASE 4.1: Cache the data after successful ingestion
        cache_key = f"av:realtime_options:{symbol}"
        self.cache.set(cache_key, api_response, ttl=30)
        print(f"✓ Data cached for {symbol}")
        
        print(f"✓ Ingestion complete: {records_inserted} inserted, {records_updated} updated")
        return records_inserted + records_updated
    
    def ingest_historical_options(self, api_response, symbol, data_date=None):
        """
        Ingest HISTORICAL_OPTIONS data into database
        Phase 4.1: Cache after successful ingestion
        """
        if not api_response or 'data' not in api_response:
            print("No historical data to ingest")
            return 0
        
        options_data = api_response['data']
        
        # Use today's date if not specified
        if data_date is None:
            data_date = date.today()
        
        print(f"Processing {len(options_data)} historical option contracts for {data_date}...")
        
        records_inserted = 0
        records_updated = 0
        
        with self.engine.connect() as conn:
            for contract in options_data:
                try:
                    record = {
                        'contract_id': contract['contractID'],
                        'symbol': contract['symbol'],
                        'expiration': contract['expiration'],
                        'strike': self._to_decimal(contract['strike']),
                        'option_type': contract['type'],
                        'last_price': self._to_decimal(contract.get('last')),
                        'mark': self._to_decimal(contract.get('mark')),
                        'bid': self._to_decimal(contract.get('bid')),
                        'bid_size': self._to_int(contract.get('bid_size')),
                        'ask': self._to_decimal(contract.get('ask')),
                        'ask_size': self._to_int(contract.get('ask_size')),
                        'volume': self._to_int(contract.get('volume')),
                        'open_interest': self._to_int(contract.get('open_interest')),
                        'date': contract.get('date'),
                        'implied_volatility': self._to_decimal(contract.get('implied_volatility')),
                        'delta': self._to_decimal(contract.get('delta')),
                        'gamma': self._to_decimal(contract.get('gamma')),
                        'theta': self._to_decimal(contract.get('theta')),
                        'vega': self._to_decimal(contract.get('vega')),
                        'rho': self._to_decimal(contract.get('rho')),
                        'data_date': data_date
                    }
                    
                    # Check if this contract+date exists
                    existing = conn.execute(
                        text("""SELECT id FROM av_historical_options 
                               WHERE contract_id = :contract_id AND data_date = :data_date"""),
                        {'contract_id': record['contract_id'], 'data_date': data_date}
                    ).fetchone()
                    
                    if existing:
                        # Update existing record
                        update_query = text("""
                            UPDATE av_historical_options 
                            SET last_price = :last_price,
                                mark = :mark,
                                bid = :bid,
                                bid_size = :bid_size,
                                ask = :ask,
                                ask_size = :ask_size,
                                volume = :volume,
                                open_interest = :open_interest,
                                implied_volatility = :implied_volatility,
                                delta = :delta,
                                gamma = :gamma,
                                theta = :theta,
                                vega = :vega,
                                rho = :rho
                            WHERE contract_id = :contract_id AND data_date = :data_date
                        """)
                        conn.execute(update_query, record)
                        records_updated += 1
                    else:
                        # Insert new record
                        insert_query = text("""
                            INSERT INTO av_historical_options 
                            (contract_id, symbol, expiration, strike, option_type,
                             last_price, mark, bid, bid_size, ask, ask_size,
                             volume, open_interest, date, implied_volatility,
                             delta, gamma, theta, vega, rho, data_date)
                            VALUES 
                            (:contract_id, :symbol, :expiration, :strike, :option_type,
                             :last_price, :mark, :bid, :bid_size, :ask, :ask_size,
                             :volume, :open_interest, :date, :implied_volatility,
                             :delta, :gamma, :theta, :vega, :rho, :data_date)
                        """)
                        conn.execute(insert_query, record)
                        records_inserted += 1
                        
                except Exception as e:
                    print(f"Error processing historical contract: {e}")
                    continue
            
            conn.commit()
        
        # PHASE 4.1: Cache the data after successful ingestion
        date_str = str(data_date) if data_date else 'latest'
        cache_key = f"av:historical_options:{symbol}:{date_str}"
        self.cache.set(cache_key, api_response, ttl=86400)  # 24 hours
        print(f"✓ Historical data cached for {symbol}")
        
        print(f"✓ Historical ingestion complete: {records_inserted} inserted, {records_updated} updated")
        return records_inserted + records_updated

    def ingest_rsi_data(self, api_response, symbol, interval, time_period):
        """
        Ingest RSI indicator data into database
        Phase 5.1: Technical indicator ingestion
        
        Args:
            api_response: Raw API response from Alpha Vantage
            symbol: Stock symbol
            interval: Time interval (from API call)
            time_period: RSI period (from API call)
        """
        if not api_response or 'Technical Analysis: RSI' not in api_response:
            print(f"No RSI data to ingest for {symbol}")
            return 0
        
        rsi_data = api_response['Technical Analysis: RSI']
        print(f"Processing {len(rsi_data)} RSI data points for {symbol}...")
        
        records_inserted = 0
        records_updated = 0
        
        with self.engine.connect() as conn:
            for timestamp_str, rsi_dict in rsi_data.items():
                try:
                    # Parse timestamp (format: '2025-08-15 20:00')
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                    
                    # Extract RSI value from nested dict {'RSI': '36.4294'}
                    rsi_value = self._to_decimal(rsi_dict.get('RSI'))
                    
                    if rsi_value is None:
                        continue
                    
                    # Check if this data point exists
                    existing = conn.execute(
                        text("""
                            SELECT id FROM av_rsi 
                            WHERE symbol = :symbol 
                            AND timestamp = :timestamp 
                            AND interval = :interval 
                            AND time_period = :time_period
                        """),
                        {
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'interval': interval,
                            'time_period': time_period
                        }
                    ).fetchone()
                    
                    if existing:
                        # Update existing record
                        conn.execute(
                            text("""
                                UPDATE av_rsi 
                                SET rsi = :rsi,
                                    updated_at = :updated_at
                                WHERE symbol = :symbol 
                                AND timestamp = :timestamp 
                                AND interval = :interval 
                                AND time_period = :time_period
                            """),
                            {
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'interval': interval,
                                'time_period': time_period,
                                'rsi': rsi_value,
                                'updated_at': datetime.now()
                            }
                        )
                        records_updated += 1
                    else:
                        # Insert new record
                        conn.execute(
                            text("""
                                INSERT INTO av_rsi 
                                (symbol, timestamp, rsi, interval, time_period, created_at, updated_at)
                                VALUES 
                                (:symbol, :timestamp, :rsi, :interval, :time_period, :created_at, :updated_at)
                            """),
                            {
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'rsi': rsi_value,
                                'interval': interval,
                                'time_period': time_period,
                                'created_at': datetime.now(),
                                'updated_at': datetime.now()
                            }
                        )
                        records_inserted += 1
                    
                    # Process in batches to avoid overwhelming
                    if (records_inserted + records_updated) % 1000 == 0:
                        conn.commit()
                        print(f"  Processed {records_inserted + records_updated} records...")
                        
                except Exception as e:
                    print(f"Error processing RSI timestamp {timestamp_str}: {e}")
                    continue
            
            conn.commit()
        
        # Cache the data after successful ingestion
        cache_key = f"av:rsi:{symbol}:{interval}_{time_period}"
        self.cache.set(cache_key, api_response, ttl=60)
        print(f"✓ RSI data cached for {symbol}")
        
        total_records = records_inserted + records_updated
        print(f"✓ RSI ingestion complete: {records_inserted} inserted, {records_updated} updated")
        return total_records

    def ingest_macd_data(self, api_response, symbol, interval,fastperiod, slowperiod, signalperiod, series_type):
        """
        Ingest MACD indicator data into database
        Phase 5.2: Technical indicator with 3 values per timestamp
        
        Args:
            api_response: Raw API response from Alpha Vantage
            symbol: Stock symbol (REQUIRED)
            interval: Time interval (REQUIRED)
            fastperiod: Fast EMA period (REQUIRED)
            slowperiod: Slow EMA period (REQUIRED)
            signalperiod: Signal EMA period (REQUIRED)
            series_type: Price type used (REQUIRED)
        """
        if not api_response or 'Technical Analysis: MACD' not in api_response:
            print(f"No MACD data to ingest for {symbol}")
            return 0
        
        macd_data = api_response['Technical Analysis: MACD']
        print(f"Processing {len(macd_data)} MACD data points for {symbol}...")
        
        records_inserted = 0
        records_updated = 0
        
        with self.engine.connect() as conn:
            for timestamp_str, macd_dict in macd_data.items():
                try:
                    # Parse timestamp (format: '2025-08-15 20:00')
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                    
                    # Extract all 3 MACD values
                    macd_value = self._to_decimal(macd_dict.get('MACD'))
                    signal_value = self._to_decimal(macd_dict.get('MACD_Signal'))
                    hist_value = self._to_decimal(macd_dict.get('MACD_Hist'))
                    
                    if macd_value is None or signal_value is None or hist_value is None:
                        continue
                    
                    # Check if this data point exists
                    existing = conn.execute(
                        text("""
                            SELECT id FROM av_macd 
                            WHERE symbol = :symbol 
                            AND timestamp = :timestamp 
                            AND interval = :interval 
                            AND fastperiod = :fastperiod
                            AND slowperiod = :slowperiod
                            AND signalperiod = :signalperiod
                        """),
                        {
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'interval': interval,
                            'fastperiod': fastperiod,
                            'slowperiod': slowperiod,
                            'signalperiod': signalperiod
                        }
                    ).fetchone()
                    
                    if existing:
                        # Update existing record
                        conn.execute(
                            text("""
                                UPDATE av_macd 
                                SET macd = :macd,
                                    macd_signal = :macd_signal,
                                    macd_hist = :macd_hist,
                                    series_type = :series_type,
                                    updated_at = :updated_at
                                WHERE symbol = :symbol 
                                AND timestamp = :timestamp 
                                AND interval = :interval 
                                AND fastperiod = :fastperiod
                                AND slowperiod = :slowperiod
                                AND signalperiod = :signalperiod
                            """),
                            {
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'interval': interval,
                                'fastperiod': fastperiod,
                                'slowperiod': slowperiod,
                                'signalperiod': signalperiod,
                                'series_type': series_type,
                                'macd': macd_value,
                                'macd_signal': signal_value,
                                'macd_hist': hist_value,
                                'updated_at': datetime.now()
                            }
                        )
                        records_updated += 1
                    else:
                        # Insert new record
                        conn.execute(
                            text("""
                                INSERT INTO av_macd 
                                (symbol, timestamp, macd, macd_signal, macd_hist, 
                                interval, fastperiod, slowperiod, signalperiod, series_type,
                                created_at, updated_at)
                                VALUES 
                                (:symbol, :timestamp, :macd, :macd_signal, :macd_hist,
                                :interval, :fastperiod, :slowperiod, :signalperiod, :series_type,
                                :created_at, :updated_at)
                            """),
                            {
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'macd': macd_value,
                                'macd_signal': signal_value,
                                'macd_hist': hist_value,
                                'interval': interval,
                                'fastperiod': fastperiod,
                                'slowperiod': slowperiod,
                                'signalperiod': signalperiod,
                                'series_type': series_type,
                                'created_at': datetime.now(),
                                'updated_at': datetime.now()
                            }
                        )
                        records_inserted += 1
                    
                    # Process in batches to avoid overwhelming
                    if (records_inserted + records_updated) % 1000 == 0:
                        conn.commit()
                        print(f"  Processed {records_inserted + records_updated} records...")
                        
                except Exception as e:
                    print(f"Error processing MACD timestamp {timestamp_str}: {e}")
                    continue
            
            conn.commit()
        
        # Cache the data after successful ingestion
        cache_key = f"av:macd:{symbol}:{interval}_{fastperiod}_{slowperiod}_{signalperiod}"
        self.cache.set(cache_key, api_response, ttl=60)
        print(f"✓ MACD data cached for {symbol}")
        
        total_records = records_inserted + records_updated
        print(f"✓ MACD ingestion complete: {records_inserted} inserted, {records_updated} updated")
        return total_records

    def ingest_bbands_data(self, api_response, symbol, interval, time_period,
                        nbdevup, nbdevdn, matype, series_type):
        """
        Ingest Bollinger Bands indicator data into database
        Phase 5.3: Technical indicator with 3 bands per timestamp
        
        Args:
            api_response: Raw API response from Alpha Vantage
            symbol: Stock symbol (REQUIRED)
            interval: Time interval (REQUIRED)
            time_period: Number of data points (REQUIRED)
            nbdevup: Upper band std deviations (REQUIRED)
            nbdevdn: Lower band std deviations (REQUIRED)
            matype: Moving average type (REQUIRED)
            series_type: Price type used (REQUIRED)
        """
        if not api_response or 'Technical Analysis: BBANDS' not in api_response:
            print(f"No BBANDS data to ingest for {symbol}")
            return 0
        
        bbands_data = api_response['Technical Analysis: BBANDS']
        print(f"Processing {len(bbands_data)} BBANDS data points for {symbol}...")
        
        records_inserted = 0
        records_updated = 0
        
        with self.engine.connect() as conn:
            for timestamp_str, bbands_dict in bbands_data.items():
                try:
                    # Parse timestamp (format: '2025-08-15 20:00')
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                    
                    # Extract all 3 band values
                    upper_band = self._to_decimal(bbands_dict.get('Real Upper Band'))
                    middle_band = self._to_decimal(bbands_dict.get('Real Middle Band'))
                    lower_band = self._to_decimal(bbands_dict.get('Real Lower Band'))
                    
                    if upper_band is None or middle_band is None or lower_band is None:
                        continue
                    
                    # Check if this data point exists
                    existing = conn.execute(
                        text("""
                            SELECT id FROM av_bbands 
                            WHERE symbol = :symbol 
                            AND timestamp = :timestamp 
                            AND interval = :interval 
                            AND time_period = :time_period
                            AND nbdevup = :nbdevup
                            AND nbdevdn = :nbdevdn
                            AND matype = :matype
                        """),
                        {
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'interval': interval,
                            'time_period': time_period,
                            'nbdevup': nbdevup,
                            'nbdevdn': nbdevdn,
                            'matype': matype
                        }
                    ).fetchone()
                    
                    if existing:
                        # Update existing record
                        conn.execute(
                            text("""
                                UPDATE av_bbands 
                                SET upper_band = :upper_band,
                                    middle_band = :middle_band,
                                    lower_band = :lower_band,
                                    series_type = :series_type,
                                    updated_at = :updated_at
                                WHERE symbol = :symbol 
                                AND timestamp = :timestamp 
                                AND interval = :interval 
                                AND time_period = :time_period
                                AND nbdevup = :nbdevup
                                AND nbdevdn = :nbdevdn
                                AND matype = :matype
                            """),
                            {
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'interval': interval,
                                'time_period': time_period,
                                'nbdevup': nbdevup,
                                'nbdevdn': nbdevdn,
                                'matype': matype,
                                'series_type': series_type,
                                'upper_band': upper_band,
                                'middle_band': middle_band,
                                'lower_band': lower_band,
                                'updated_at': datetime.now()
                            }
                        )
                        records_updated += 1
                    else:
                        # Insert new record
                        conn.execute(
                            text("""
                                INSERT INTO av_bbands 
                                (symbol, timestamp, upper_band, middle_band, lower_band,
                                interval, time_period, nbdevup, nbdevdn, matype, series_type,
                                created_at, updated_at)
                                VALUES 
                                (:symbol, :timestamp, :upper_band, :middle_band, :lower_band,
                                :interval, :time_period, :nbdevup, :nbdevdn, :matype, :series_type,
                                :created_at, :updated_at)
                            """),
                            {
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'upper_band': upper_band,
                                'middle_band': middle_band,
                                'lower_band': lower_band,
                                'interval': interval,
                                'time_period': time_period,
                                'nbdevup': nbdevup,
                                'nbdevdn': nbdevdn,
                                'matype': matype,
                                'series_type': series_type,
                                'created_at': datetime.now(),
                                'updated_at': datetime.now()
                            }
                        )
                        records_inserted += 1
                    
                    # Process in batches to avoid overwhelming
                    if (records_inserted + records_updated) % 1000 == 0:
                        conn.commit()
                        print(f"  Processed {records_inserted + records_updated} records...")
                        
                except Exception as e:
                    print(f"Error processing BBANDS timestamp {timestamp_str}: {e}")
                    continue
            
            conn.commit()
        
        # Cache the data after successful ingestion
        cache_key = f"av:bbands:{symbol}:{interval}_{time_period}_{nbdevup}_{nbdevdn}_{matype}"
        self.cache.set(cache_key, api_response, ttl=60)
        print(f"✓ BBANDS data cached for {symbol}")
        
        total_records = records_inserted + records_updated
        print(f"✓ BBANDS ingestion complete: {records_inserted} inserted, {records_updated} updated")
        return total_records

    def ingest_vwap_data(self, vwap_data, symbol, interval):
        """
        Ingest VWAP data into database
        
        Args:
            vwap_data: API response from Alpha Vantage
            symbol: Stock symbol
            interval: Time interval (1min, 5min, etc.)
        
        Returns:
            int: Number of records processed
        """
        if not vwap_data or 'Technical Analysis: VWAP' not in vwap_data:
            print(f"No VWAP data to ingest for {symbol}")
            return 0
        
        # Extract VWAP values
        vwap_values = vwap_data['Technical Analysis: VWAP']
        
        # Prepare records for insertion
        records = []
        for timestamp_str, value_dict in vwap_values.items():
            # Parse timestamp - VWAP timestamps don't have seconds
            try:
                # Try with seconds first
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Fall back to without seconds
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
            
            # Extract VWAP value from nested dict {'VWAP': 'value'}
            vwap_value = float(value_dict['VWAP'])
            
            records.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'vwap': vwap_value,
                'interval': interval,
                'updated_at': datetime.now()
            })
        
        # Batch insert into database
        if records:
            # Process in batches of 1000
            batch_size = 1000
            total_inserted = 0
            
            with self.engine.begin() as conn:
                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    
                    # Use INSERT ... ON CONFLICT DO UPDATE
                    stmt = text("""
                        INSERT INTO av_vwap (symbol, timestamp, vwap, interval, updated_at)
                        VALUES (:symbol, :timestamp, :vwap, :interval, :updated_at)
                        ON CONFLICT (symbol, timestamp, interval)
                        DO UPDATE SET 
                            vwap = EXCLUDED.vwap,
                            updated_at = EXCLUDED.updated_at
                    """)
                    
                    conn.execute(stmt, batch)
                    total_inserted += len(batch)
            
            print(f"Ingested {total_inserted} VWAP records for {symbol} ({interval})")
            
            # Update cache with fresh data
            cache_key = f"av:vwap:{symbol}:{interval}"
            cache_ttl = self.config.av_config['endpoints']['vwap'].get('cache_ttl', 60)
            self.cache.set(cache_key, vwap_data, cache_ttl)
            
            return total_inserted
        
        return 0

    def ingest_atr_data(self, api_response, symbol, interval=None, time_period=None):
            """
            Ingest ATR (Average True Range) indicator data into database
            Phase 5.5 - Day 22: Volatility indicator ingestion
            
            ATR is different from other indicators:
            - Uses DATE not TIMESTAMP (daily data)
            - Single value per day (volatility in price units)
            - Updates less frequently (daily market close)
            
            Args:
                api_response: Raw API response from Alpha Vantage
                symbol: Stock symbol
                interval: Time interval (from config if not provided)
                time_period: ATR calculation period (from config if not provided)
            
            Returns:
                Number of records processed
            """
            # Get defaults from config - NO HARDCODING!
            if interval is None:
                interval = self.config.av_config['endpoints']['atr']['default_params']['interval']
            if time_period is None:
                time_period = self.config.av_config['endpoints']['atr']['default_params']['time_period']
            
            if not api_response or 'Technical Analysis: ATR' not in api_response:
                print(f"No ATR data to ingest for {symbol}")
                return 0
            
            atr_data = api_response['Technical Analysis: ATR']
            print(f"Processing {len(atr_data)} ATR data points for {symbol}...")
            
            records_inserted = 0
            records_updated = 0
            batch = []
            
            with self.engine.connect() as conn:
                # Process each date's ATR value
                for date_str, atr_dict in atr_data.items():
                    try:
                        # Parse date (not datetime since ATR is daily)
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        # Extract ATR value
                        atr_value = self._to_decimal(atr_dict.get('ATR'))
                        
                        if atr_value is None:
                            continue
                        
                        # Check if record exists
                        check_query = text("""
                            SELECT id FROM av_atr 
                            WHERE symbol = :symbol 
                            AND timestamp = :timestamp 
                            AND time_period = :time_period
                        """)
                        
                        result = conn.execute(check_query, {
                            'symbol': symbol,
                            'timestamp': date_obj,
                            'time_period': time_period
                        })
                        existing = result.fetchone()
                        
                        if existing:
                            # Update existing record
                            update_query = text("""
                                UPDATE av_atr 
                                SET atr = :atr,
                                    interval = :interval,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE symbol = :symbol 
                                AND timestamp = :timestamp 
                                AND time_period = :time_period
                            """)
                            
                            conn.execute(update_query, {
                                'symbol': symbol,
                                'timestamp': date_obj,
                                'atr': atr_value,
                                'interval': interval,
                                'time_period': time_period
                            })
                            records_updated += 1
                        else:
                            # Prepare for batch insert
                            batch.append({
                                'symbol': symbol,
                                'timestamp': date_obj,
                                'atr': atr_value,
                                'interval': interval,
                                'time_period': time_period
                            })
                            
                            # Insert in batches of 1000
                            if len(batch) >= 1000:
                                insert_query = text("""
                                    INSERT INTO av_atr 
                                    (symbol, timestamp, atr, interval, time_period)
                                    VALUES (:symbol, :timestamp, :atr, :interval, :time_period)
                                """)
                                conn.execute(insert_query, batch)
                                conn.commit()
                                records_inserted += len(batch)
                                print(f"  Inserted batch of {len(batch)} ATR records")
                                batch = []
                    
                    except Exception as e:
                        print(f"  Error processing ATR date {date_str}: {str(e)[:100]}")
                        continue
                
                # Insert remaining batch
                if batch:
                    insert_query = text("""
                        INSERT INTO av_atr 
                        (symbol, timestamp, atr, interval, time_period)
                        VALUES (:symbol, :timestamp, :atr, :interval, :time_period)
                    """)
                    conn.execute(insert_query, batch)
                    conn.commit()
                    records_inserted += len(batch)
                    print(f"  Inserted final batch of {len(batch)} ATR records")
            
            total_records = records_inserted + records_updated
            
            # Cache the response after successful ingestion
            cache_key = f"av:atr:{symbol}:{interval}_{time_period}"
            cache_ttl = 300  # 5 minutes for daily data
            self.cache.set(cache_key, api_response, ttl=cache_ttl)
            
            print(f"✓ ATR ingestion complete for {symbol}:")
            print(f"  - Inserted: {records_inserted}")
            print(f"  - Updated: {records_updated}")
            print(f"  - Total: {total_records}")
            
            # Show latest ATR value for context
            if atr_data:
                latest_date = list(atr_data.keys())[0]
                latest_atr = atr_data[latest_date].get('ATR')
                print(f"  - Latest ATR ({latest_date}): {latest_atr} (${latest_atr} daily range)")
            
            return total_records    

    def ingest_ibkr_bar(self, symbol, timestamp, open_, high, low, close, volume, vwap, count, bar_size='5sec'):
        """
        Ingest IBKR bar data
        Phase 3.5: Store real-time bar data
        """
        table_map = {
            '5sec': 'ibkr_bars_5sec',
            '1min': 'ibkr_bars_1min', 
            '5min': 'ibkr_bars_5min'
        }
        
        table = table_map.get(bar_size, 'ibkr_bars_5sec')
        
        try:
            with self.engine.connect() as conn:
                query = text(f"""
                    INSERT INTO {table} 
                    (symbol, timestamp, open, high, low, close, volume, vwap, bar_count)
                    VALUES 
                    (:symbol, :timestamp, :open, :high, :low, :close, :volume, :vwap, :bar_count)
                    ON CONFLICT (symbol, timestamp) 
                    DO UPDATE SET
                        close = :close,
                        volume = :volume,
                        vwap = :vwap
                """)
                
                conn.execute(query, {
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'open': Decimal(str(open_)),
                    'high': Decimal(str(high)),
                    'low': Decimal(str(low)),
                    'close': Decimal(str(close)),
                    'volume': volume,
                    'vwap': Decimal(str(vwap)) if vwap else None,
                    'bar_count': count
                })
                conn.commit()
                
        except Exception as e:
            print(f"Error storing bar data: {e}")
    
    def ingest_ibkr_quote(self, symbol, bid, bid_size, ask, ask_size, last, last_size, volume):
        """
        Ingest IBKR quote data
        Phase 3.5: Store real-time quotes
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO ibkr_quotes
                    (symbol, timestamp, bid, bid_size, ask, ask_size, last, last_size, volume)
                    VALUES
                    (:symbol, :timestamp, :bid, :bid_size, :ask, :ask_size, :last, :last_size, :volume)
                """)
                
                conn.execute(query, {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'bid': Decimal(str(bid)) if bid > 0 else None,
                    'bid_size': bid_size if bid_size > 0 else None,
                    'ask': Decimal(str(ask)) if ask > 0 else None,
                    'ask_size': ask_size if ask_size > 0 else None,
                    'last': Decimal(str(last)) if last > 0 else None,
                    'last_size': last_size if last_size > 0 else None,
                    'volume': volume if volume > 0 else None
                })
                conn.commit()
                
        except Exception as e:
            print(f"Error storing quote data: {e}")

    def _to_decimal(self, value):
        """Convert string to Decimal, handling None and logging issues"""
        if value is None:
            return None
        
        if value == '':
            # Empty string is suspicious - log it
            print(f"  ⚠️ Warning: Empty string found where number expected")
            return None
        
        try:
            return Decimal(str(value))
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to convert '{value}' to Decimal: {e}")
            return None
    
    def _to_int(self, value):
        """Convert string to int, handling None and logging issues"""
        if value is None:
            return None
            
        if value == '':
            # Empty string is suspicious for integers
            print(f"  ⚠️ Warning: Empty string found where integer expected")
            return None
        
        try:
            return int(value)
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to convert '{value}' to int: {e}")
            return None