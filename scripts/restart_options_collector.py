#!/usr/bin/env python3
"""
Restart the data collector with options-optimized configuration
This script ensures the data collector prioritizes options data
"""
import subprocess
import sys
import time
import os

def kill_existing_collectors():
    """Kill any existing data collector processes"""
    print("🛑 Stopping existing data collectors...")
    try:
        # Find and kill processes
        result = subprocess.run(
            ["pgrep", "-f", "start_data_collector.py"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print(f"   Killing PID {pid}")
                subprocess.run(["kill", "-9", pid])
            time.sleep(2)
            print("✅ Existing collectors stopped")
        else:
            print("   No existing collectors found")
    except Exception as e:
        print(f"⚠️  Error stopping collectors: {e}")

def start_new_collector():
    """Start the optimized data collector"""
    print("\n🚀 Starting OPTIONS-OPTIMIZED data collector...")
    print("   • Options: Every 30 seconds (24/7)")
    print("   • Indicators: Every 30 minutes (reduced)")
    print("   • Historical Options: Every 4 hours")
    print("\n" + "="*60)
    
    # Set environment to ensure proper configuration
    env = os.environ.copy()
    env['PYTHONPATH'] = '/Users/michaelmerrick/AlphaTrader'
    
    # Start the collector
    try:
        subprocess.run([
            sys.executable,
            "/Users/michaelmerrick/AlphaTrader/scripts/startup/start_data_collector.py"
        ], env=env)
    except KeyboardInterrupt:
        print("\n\n✋ Collector stopped by user")
    except Exception as e:
        print(f"\n❌ Error running collector: {e}")

if __name__ == "__main__":
    print("="*60)
    print("🎯 OPTIONS TRADING DATA COLLECTOR RESTART")
    print("="*60)
    
    kill_existing_collectors()
    start_new_collector()