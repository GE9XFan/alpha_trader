#!/usr/bin/env python3
"""Test IBKR feeds and save whatever we get"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import get_config_manager
from src.foundation.logger import get_logger
from ib_insync import IB, Stock


def test_ibkr():
    config = get_config_manager()
    logger = get_logger('IBKRTest')
    
    response_dir = Path('data/api_responses/ibkr')
    response_dir.mkdir(parents=True, exist_ok=True)
    
    def serialize_ib_object(obj):
        """Serialize ib_insync objects properly"""
        if hasattr(obj, '_asdict'):
            return obj._asdict()
        elif hasattr(obj, '__slots__'):
            return {slot: getattr(obj, slot, None) for slot in obj.__slots__}
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    ib = IB()
    
    try:
        # Connect
        ib.connect(
            host=config.ibkr_config.get('host', '127.0.0.1'),
            port=config.ibkr_config.get('port', 7497),
            clientId=config.ibkr_config.get('client_id', 1)
        )
        
        logger.info("Connected to IBKR")
        time.sleep(2)  # Let connection settle
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Test 1: Account info
        try:
            accounts = ib.managedAccounts()
            with open(response_dir / f'accounts_{timestamp}.json', 'w') as f:
                json.dump({'accounts': accounts}, f, indent=2)
            logger.info(f"Accounts: {accounts}")
        except Exception as e:
            logger.error(f"Account info failed: {e}")
        
        # Test 2: Account summary
        try:
            summary = ib.accountSummary()
            if summary:
                summary_data = [serialize_ib_object(s) for s in summary]
                with open(response_dir / f'summary_{timestamp}.json', 'w') as f:
                    json.dump(summary_data, f, indent=2, default=str)
                logger.info(f"Summary items: {len(summary)}")
        except Exception as e:
            logger.error(f"Account summary failed: {e}")
        
        # Test 3: Positions
        try:
            positions = ib.positions()
            if positions:
                position_data = [serialize_ib_object(p) for p in positions]
                with open(response_dir / f'positions_{timestamp}.json', 'w') as f:
                    json.dump(position_data, f, indent=2, default=str)
                logger.info(f"Positions: {len(positions)}")
            else:
                logger.info("No positions found")
        except Exception as e:
            logger.error(f"Positions failed: {e}")
        
        # Test 4: Contract details
        try:
            contract = Stock('AAPL', 'SMART', 'USD')
            details = ib.reqContractDetails(contract)
            
            if details:
                detail_data = serialize_ib_object(details[0])
                with open(response_dir / f'contract_details_{timestamp}.json', 'w') as f:
                    json.dump(detail_data, f, indent=2, default=str)
                logger.info("Contract details retrieved")
        except Exception as e:
            logger.error(f"Contract details failed: {e}")
        
        # Test 5: Historical bars
        try:
            bars = ib.reqHistoricalData(
                contract, '', '1 D', '5 secs', 'TRADES', True
            )
            
            if bars:
                bar_data = [serialize_ib_object(b) for b in bars[-5:]]
                with open(response_dir / f'historical_5sec_{timestamp}.json', 'w') as f:
                    json.dump(bar_data, f, indent=2, default=str)
                logger.info(f"Historical bars: {len(bars)}")
        except Exception as e:
            logger.error(f"Historical bars failed: {e}")
        
        # Test 6: Market data
        try:
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(5)  # Wait for data
            
            ticker_data = serialize_ib_object(ticker)
            with open(response_dir / f'ticker_{timestamp}.json', 'w') as f:
                json.dump(ticker_data, f, indent=2, default=str)
            
            logger.info(f"Ticker data: bid={ticker.bid} ask={ticker.ask}")
            ib.cancelMktData(contract)
        except Exception as e:
            logger.error(f"Market data failed: {e}")
        
        # Test 7: Real-time bars
        try:
            rtbars = ib.reqRealTimeBars(contract, 5, 'TRADES', False)
            ib.sleep(10)  # Wait for bars
            
            if rtbars:
                rtbar_data = [serialize_ib_object(b) for b in rtbars]
                with open(response_dir / f'realtime_bars_{timestamp}.json', 'w') as f:
                    json.dump(rtbar_data, f, indent=2, default=str)
                logger.info(f"Real-time bars: {len(rtbars)}")
            
            ib.cancelRealTimeBars(rtbars)
        except Exception as e:
            logger.error(f"Real-time bars failed: {e}")
        
        logger.info(f"IBKR test complete - data saved to {response_dir}")
        
    except Exception as e:
        logger.error(f"IBKR connection failed: {e}")
    
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    test_ibkr()