#!/usr/bin/env python3
"""
Test Alpha Vantage API directly to verify Greeks parameter
"""
import aiohttp
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('AV_API_KEY')
BASE_URL = "https://www.alphavantage.co/query"

async def test_api_with_greeks():
    """Test API WITH require_greeks=true"""
    params = {
        'function': 'REALTIME_OPTIONS',
        'symbol': 'SPY',
        'apikey': API_KEY,
        'require_greeks': 'true'  # CRITICAL: This must be included!
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(BASE_URL, params=params) as resp:
            data = await resp.json()
            
            # Check first few contracts for Greeks
            print("\n✅ WITH require_greeks=true:")
            print(f"Response keys: {list(data.keys())[:5]}")
            
            # Find first contract with data
            for key in data:
                if key != 'Meta Data' and isinstance(data[key], list) and len(data[key]) > 0:
                    contract = data[key][0]
                    print(f"\nFirst contract keys: {list(contract.keys())}")
                    print(f"Sample contract: {json.dumps(contract, indent=2)[:500]}")
                    
                    # Check for Greeks
                    has_greeks = any(k.lower() in ['delta', 'gamma', 'theta', 'vega', 'rho'] 
                                   for k in contract.keys())
                    print(f"\n🎯 Greeks found: {has_greeks}")
                    
                    if has_greeks:
                        print("Greek values:")
                        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                            for k in contract.keys():
                                if greek in k.lower():
                                    print(f"  {k}: {contract[k]}")
                    break

async def test_api_without_greeks():
    """Test API WITHOUT require_greeks parameter"""
    params = {
        'function': 'REALTIME_OPTIONS',
        'symbol': 'SPY',
        'apikey': API_KEY
        # NOTE: No require_greeks parameter
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(BASE_URL, params=params) as resp:
            data = await resp.json()
            
            # Check first few contracts
            print("\n❌ WITHOUT require_greeks:")
            print(f"Response keys: {list(data.keys())[:5]}")
            
            # Find first contract with data
            for key in data:
                if key != 'Meta Data' and isinstance(data[key], list) and len(data[key]) > 0:
                    contract = data[key][0]
                    print(f"\nFirst contract keys: {list(contract.keys())}")
                    print(f"Sample contract: {json.dumps(contract, indent=2)[:500]}")
                    
                    # Check for Greeks
                    has_greeks = any(k.lower() in ['delta', 'gamma', 'theta', 'vega', 'rho'] 
                                   for k in contract.keys())
                    print(f"\n🎯 Greeks found: {has_greeks}")
                    break

async def main():
    print("="*70)
    print("TESTING ALPHA VANTAGE GREEKS PARAMETER")
    print("="*70)
    print(f"API Key: {API_KEY[:8]}...")
    
    # Test WITHOUT Greeks parameter
    await test_api_without_greeks()
    
    # Test WITH Greeks parameter
    await test_api_with_greeks()
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("The require_greeks=true parameter MUST be included in the API request")
    print("to receive Greeks in the response!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())