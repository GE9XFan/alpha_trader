"""Data ingestion module - Phase 2 with multiple APIs"""

import sys
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.foundation.config_manager import ConfigManager


class DataIngestion:
    def __init__(self):
        self.config = ConfigManager()
        self.engine = create_engine(self.config.database_url)
    
    def ingest_options_data(self, api_response, symbol):
        """
        Ingest REALTIME_OPTIONS data into database
        Phase 1: Simple version - parse, validate, store
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
        
        print(f"✓ Ingestion complete: {records_inserted} inserted, {records_updated} updated")
        return records_inserted + records_updated
    
    def ingest_historical_options(self, api_response, symbol, data_date=None):
        """
        Ingest HISTORICAL_OPTIONS data into database
        Phase 2.3: Second API endpoint
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
        
        print(f"✓ Historical ingestion complete: {records_inserted} inserted, {records_updated} updated")
        return records_inserted + records_updated
    
    def _to_decimal(self, value):
        """Convert string to Decimal, handling None"""
        if value is None or value == '':
            return None
        try:
            return Decimal(str(value))
        except:
            return None
    
    def _to_int(self, value):
        """Convert string to int, handling None"""
        if value is None or value == '':
            return None
        try:
            return int(value)
        except:
            return None