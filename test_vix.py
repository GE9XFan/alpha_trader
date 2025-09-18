import asyncio
import aiohttp
import csv
import io

async def test_cboe():
    url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX1D_History.csv'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
            if rows:
                latest = rows[-1]
                print(f"Latest VIX1D: {latest}")
                return float(latest.get('close', 0))
    return None

if __name__ == "__main__":
    result = asyncio.run(test_cboe())
    print(f"VIX1D value: {result}")
