from types import SimpleNamespace

import pytest

from src.ibkr_processor import IBKRDataProcessor


def _build_processor():
    config = {
        'modules': {
            'data_ingestion': {
                'store_ttls': {
                    'market_data': 60,
                    'order_book': 60,
                    'trades': 3600,
                }
            }
        }
    }
    return IBKRDataProcessor(config)


class DummyTicker:
    def __init__(self, symbol, bids=None, asks=None, bid=None, ask=None, last=None, volume=None, vwap=None):
        self.contract = SimpleNamespace(symbol=symbol)
        self.domBids = bids
        self.domAsks = asks
        self.bid = bid
        self.ask = ask
        self.last = last
        self.volume = volume
        self.vwap = vwap


class DummyLevel:
    def __init__(self, price, size, market_maker):
        self.price = price
        self.size = size
        self.marketMaker = market_maker


def test_depth_normalization_and_aggregation():
    processor = _build_processor()
    bids = [DummyLevel(100.12, 10, 'ISLAND')]
    asks = [DummyLevel(100.32, 15, 'BATS')]
    ticker = DummyTicker('SPY', bids=bids, asks=asks)

    book = processor.update_depth_book('SPY', 'SMART', ticker)
    assert book['bids'][0]['venue'] == 'NSDQ'
    assert book['asks'][0]['venue'] == 'BATS'

    processor.last_trade['SPY'] = 100.25
    aggregated = processor.compute_aggregated_tob('SPY')
    assert aggregated is not None
    assert pytest.approx(aggregated['bid'], rel=1e-6) == 100.12
    assert pytest.approx(aggregated['ask'], rel=1e-6) == 100.32
    assert aggregated['bid_exchange'] == 'SMART'
    assert aggregated['ask_exchange'] == 'SMART'
    assert aggregated['last'] == 100.25


def test_prepare_trade_storage_and_quote_cache():
    processor = _build_processor()
    trade = {'price': 250.55, 'time': 1_694_000_000_000}
    ticker = DummyTicker('AAPL', bid=250.50, ask=250.70, volume=1500, vwap=250.60)

    last_price, ticker_payload = processor.prepare_trade_storage('AAPL', trade, ticker)
    assert last_price['price'] == trade['price']
    assert ticker_payload['symbol'] == 'AAPL'
    assert ticker_payload['mid'] == pytest.approx((250.50 + 250.70) / 2, rel=1e-6)
    expected_bps = round(((250.70 - 250.50) / 250.50) * 10000, 2)
    assert ticker_payload['spread_bps'] == expected_bps
    assert processor.trades_buffer['AAPL']

    # Quote ticker should reuse cached last price when not provided
    quote = DummyTicker('AAPL', bid=250.40, ask=250.65, last=None)
    quote_data = processor.build_quote_ticker('AAPL', quote)
    assert quote_data['last'] == pytest.approx(trade['price'], rel=1e-6)
    assert quote_data['spread'] == pytest.approx(0.25, rel=1e-6)

