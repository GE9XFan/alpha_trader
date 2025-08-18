#!/usr/bin/env python3
"""
Production scheduler runner for AlphaTrader
THIS RUNS YOUR LIVE TRADING SYSTEM - HANDLE WITH CARE
"""

import sys
import signal
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.scheduler import DataScheduler

# Global scheduler for signal handling
scheduler = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received shutdown signal...")
    if scheduler and scheduler.is_running:
        print("Stopping scheduler gracefully...")
        scheduler.stop()
    print("Shutdown complete")
    sys.exit(0)

def main():
    """Main production scheduler runner"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='AlphaTrader Production Scheduler')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode (ignores market hours)')
    args = parser.parse_args()
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Kill signal
    
    print("=" * 60)
    print("ALPHATRADER PRODUCTION SCHEDULER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'TEST' if args.test else 'PRODUCTION'}")
    print("=" * 60)
    
    global scheduler
    
    try:
        # Create scheduler
        print("\nInitializing scheduler...")
        scheduler = DataScheduler(test_mode=args.test)
        
        # Start scheduler (this will connect IBKR, create jobs, etc.)
        print("Starting scheduler...")
        scheduler.start()
        
        # Get initial status
        status = scheduler.get_status()
        print(f"\n✅ Scheduler started successfully!")
        print(f"  Running: {status['running']}")
        print(f"  Jobs: {status['total_jobs']}")
        print(f"  Market Hours: {status['is_market_hours']}")
        
        if scheduler.ibkr_connected:
            print(f"  IBKR: Connected ✅")
            print(f"  Subscriptions: {len(scheduler.ibkr_subscriptions)}")
        else:
            print(f"  IBKR: Not connected (will connect at market open)")
        
        print("\n" + "=" * 60)
        print("SCHEDULER RUNNING")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        # Keep the main thread alive
        while True:
            time.sleep(60)  # Sleep for 1 minute
            
            # Optional: Print periodic status updates
            if datetime.now().minute % 15 == 0:  # Every 15 minutes
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Scheduler healthy - {status['total_jobs']} jobs running")
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
        signal_handler(None, None)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if scheduler:
            try:
                scheduler.stop()
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()