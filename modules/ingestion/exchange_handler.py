#!/usr/bin/env python3
"""
Exchange Handler Module
Manages multi-venue order book aggregation and NBBO calculation
"""

import time
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import logging

import redis
import orjson
import numpy as np
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


class ExchangeBook:
    """Efficient order book representation for a single exchange"""
    
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.bids = SortedDict(lambda x: -x)  # Reverse sort for bids (highest first)
        self.asks = SortedDict()  # Normal sort for asks (lowest first)
        self.last_update = 0
        self.sequence = 0
        self.checksum = 0
        
    def update_level(self, side: str, price: float, size: float, position: int = None):
        """Update a price level in the book"""
        
        book = self.bids if side == 'bid' else self.asks
        
        if size == 0:
            # Remove level
            if price in book:
                del book[price]
        else:
            # Add/update level
            book[price] = {
                'size': size,
                'position': position,
                'update_time': time.time_ns()
            }
            
        self.last_update = time.time_ns()
        self.sequence += 1
        
    def get_best(self, side: str) -> Optional[Tuple[float, float]]:
        """Get best bid or ask"""
        
        book = self.bids if side == 'bid' else self.asks
        
        if book:
            price = next(iter(book))
            return price, book[price]['size']
            
        return None
        
    def get_depth(self, side: str, levels: int = 10) -> List[Dict]:
        """Get top N levels of the book"""
        
        book = self.bids if side == 'bid' else self.asks
        result = []
        
        for i, (price, data) in enumerate(book.items()):
            if i >= levels:
                break
                
            result.append({
                'price': price,
                'size': data['size'],
                'position': i,
                'exchange': self.exchange
            })
            
        return result
        
    def calculate_checksum(self) -> int:
        """Calculate book checksum for integrity verification"""
        
        # Use top 10 levels for checksum
        checksum_str = ""
        
        for i, (price, data) in enumerate(self.bids.items()):
            if i >= 10:
                break
            checksum_str += f"{price:.2f}:{data['size']:.0f}:"
            
        for i, (price, data) in enumerate(self.asks.items()):
            if i >= 10:
                break
            checksum_str += f"{price:.2f}:{data['size']:.0f}:"
            
        # Simple hash for checksum
        import hashlib
        return int(hashlib.md5(checksum_str.encode()).hexdigest()[:8], 16)


class ExchangeHandler:
    """Handles multi-venue order book aggregation and NBBO calculation"""
    
    def __init__(self, config: Dict, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.symbols = config['trading']['symbols']
        
        # Complete exchange list including dark pools and retail venues
        self.exchanges = [
            # Primary exchanges
            'NYSE', 'NASDAQ', 'ARCA',
            # CBOE exchanges
            'BATS', 'BATY', 'BYX', 'BZX', 'EDGX', 'EDGA',
            # Other lit exchanges
            'IEX', 'MEMX', 'PSX', 'LTSE', 'PEARL', 'MIAX',
            # Dark pools and ATSs
            'SIGMA-X', 'LEVEL', 'CROSSFINDER', 'UBSA', 'JPMX',
            'BARX', 'MSTX', 'CITI', 'VIRTU', 'IBKR',
            # Retail venues
            'DRCTEDGE', 'CITADEL', 'G1', 'JANE'
        ]
        
        # Order books per symbol per exchange
        self.books = {
            symbol: {
                exchange: ExchangeBook(exchange)
                for exchange in self.exchanges
            } for symbol in self.symbols
        }
        
        # Consolidated books and NBBO
        self.consolidated = {
            symbol: {
                'bids': [],
                'asks': [],
                'nbbo': {
                    'bid': {'price': 0, 'size': 0, 'exchange': None},
                    'ask': {'price': float('inf'), 'size': 0, 'exchange': None}
                },
                'last_update': 0
            } for symbol in self.symbols
        }
        
        # Fragmentation metrics
        self.fragmentation = defaultdict(lambda: defaultdict(float))
        
        # Exchange routing statistics
        self.routing_stats = defaultdict(lambda: defaultdict(int))
        
        # Crossed book detection
        self.crossed_books = defaultdict(list)
        
        # Protected NBBO tracking (Reg NMS)
        self.protected_quotes = defaultdict(dict)
        
    def process_depth_update(self, symbol: str, depth, exchange_time: int, gateway_time: int):
        """Process depth update from specific exchange"""
        
        try:
            # Determine exchange from depth message
            exchange = self._identify_exchange(depth)
            
            if not exchange or exchange not in self.exchanges:
                return
                
            # Update exchange-specific book
            book = self.books[symbol][exchange]
            
            # Determine side and operation
            side = 'bid' if depth.side == 0 else 'ask'
            
            # Apply update based on operation type
            if depth.operation == 0:  # Insert
                self._insert_level(book, side, depth)
            elif depth.operation == 1:  # Update
                self._update_level(book, side, depth)
            elif depth.operation == 2:  # Delete
                self._delete_level(book, side, depth)
                
            # Rebuild consolidated book and NBBO
            self._rebuild_consolidated(symbol)
            
            # Check for crossed books
            self._check_crossed_book(symbol, exchange)
            
            # Calculate fragmentation
            self._calculate_fragmentation(symbol)
            
            # Update routing statistics
            self.routing_stats[symbol][exchange] += 1
            
            # Store in Redis
            self._store_exchange_books(symbol)
            
        except Exception as e:
            logger.error(f"Error processing depth update for {symbol}: {e}")
            
    def process_l2_update(self, symbol: str, depth):
        """Process Level 2 update with full book reconstruction"""
        
        try:
            # Identify exchange
            exchange = self._identify_exchange(depth)
            
            if not exchange:
                return
                
            book = self.books[symbol][exchange]
            
            # Process based on update type
            if hasattr(depth, 'position'):
                # Position-based update
                side = 'bid' if depth.side == 0 else 'ask'
                
                # Get current book levels
                current_levels = book.get_depth(side, 20)
                
                # Apply position-based update
                if depth.operation == 0:  # Insert at position
                    self._insert_at_position(book, side, depth.position, depth.price, depth.size)
                elif depth.operation == 1:  # Update at position
                    self._update_at_position(book, side, depth.position, depth.price, depth.size)
                elif depth.operation == 2:  # Delete at position
                    self._delete_at_position(book, side, depth.position)
                    
            else:
                # Price-based update
                side = 'bid' if depth.side == 0 else 'ask'
                book.update_level(side, depth.price, depth.size)
                
            # Update consolidated view
            self._rebuild_consolidated(symbol)
            
            # Calculate and verify checksum
            new_checksum = book.calculate_checksum()
            if book.checksum != 0 and abs(new_checksum - book.checksum) > 1000000:
                logger.warning(f"Checksum mismatch for {symbol} on {exchange}")
                # Request snapshot to resync
                self._request_snapshot(symbol, exchange)
                
            book.checksum = new_checksum
            
        except Exception as e:
            logger.error(f"Error processing L2 update for {symbol}: {e}")
            
    def _identify_exchange(self, depth) -> Optional[str]:
        """Identify exchange from depth message"""
        
        # IBKR provides exchange in marketMaker field for Level 2
        if hasattr(depth, 'marketMaker'):
            mm = depth.marketMaker
            
            # Complete market maker ID to exchange mapping
            exchange_map = {
                # Primary exchanges
                'NSDQ': 'NASDAQ', 'NYSE': 'NYSE', 'ARCA': 'ARCA',
                # CBOE exchanges
                'BATS': 'BATS', 'BATY': 'BATY', 'BYX': 'BYX', 'BZX': 'BZX',
                'EDGX': 'EDGX', 'EDGA': 'EDGA',
                # Other exchanges
                'IEX': 'IEX', 'MEMX': 'MEMX', 'PSX': 'PSX', 'LTSE': 'LTSE',
                'PERL': 'PEARL', 'MIAX': 'MIAX',
                # Dark pools
                'SGMA': 'SIGMA-X', 'LEVL': 'LEVEL', 'XFDR': 'CROSSFINDER',
                'UBSA': 'UBSA', 'JPMX': 'JPMX', 'BARX': 'BARX',
                'MSTX': 'MSTX', 'CITI': 'CITI', 'VIRT': 'VIRTU', 'IBKR': 'IBKR',
                # Retail
                'DRCT': 'DRCTEDGE', 'CDEL': 'CITADEL', 'GTS1': 'G1', 'JANE': 'JANE'
            }
            
            return exchange_map.get(mm, mm)
            
        # Try to get from exchange field
        if hasattr(depth, 'exchange'):
            return depth.exchange
            
        return None
        
    def _insert_level(self, book: ExchangeBook, side: str, depth):
        """Insert new level in book"""
        
        book.update_level(side, depth.price, depth.size, depth.position)
        
    def _update_level(self, book: ExchangeBook, side: str, depth):
        """Update existing level in book"""
        
        book.update_level(side, depth.price, depth.size, depth.position)
        
    def _delete_level(self, book: ExchangeBook, side: str, depth):
        """Delete level from book"""
        
        book.update_level(side, depth.price, 0, depth.position)
        
    def _insert_at_position(self, book: ExchangeBook, side: str, position: int, price: float, size: float):
        """Insert level at specific position"""
        
        # This requires rebuilding the book with correct positioning
        # For now, just update by price
        book.update_level(side, price, size, position)
        
    def _update_at_position(self, book: ExchangeBook, side: str, position: int, price: float, size: float):
        """Update level at specific position"""
        
        # Get current levels
        levels = book.get_depth(side, position + 1)
        
        if position < len(levels):
            old_price = levels[position]['price']
            # Remove old price if different
            if old_price != price:
                book.update_level(side, old_price, 0)
                
        # Add new price/size
        book.update_level(side, price, size, position)
        
    def _delete_at_position(self, book: ExchangeBook, side: str, position: int):
        """Delete level at specific position"""
        
        # Get current levels
        levels = book.get_depth(side, position + 1)
        
        if position < len(levels):
            price = levels[position]['price']
            book.update_level(side, price, 0)
            
    def _rebuild_consolidated(self, symbol: str):
        """Rebuild consolidated book and NBBO from all exchanges"""
        
        # Aggregate all bids and asks
        all_bids = []
        all_asks = []
        
        for exchange, book in self.books[symbol].items():
            # Get top levels from each exchange
            bid_levels = book.get_depth('bid', 10)
            ask_levels = book.get_depth('ask', 10)
            
            all_bids.extend(bid_levels)
            all_asks.extend(ask_levels)
            
        # Sort consolidated books
        all_bids.sort(key=lambda x: x['price'], reverse=True)
        all_asks.sort(key=lambda x: x['price'])
        
        # Keep top 10 levels consolidated
        self.consolidated[symbol]['bids'] = all_bids[:10]
        self.consolidated[symbol]['asks'] = all_asks[:10]
        
        # Calculate NBBO
        self._calculate_nbbo(symbol, all_bids, all_asks)
        
        # Check for locked/crossed NBBO
        self._check_locked_crossed_nbbo(symbol)
        
        self.consolidated[symbol]['last_update'] = time.time_ns()
        
    def _calculate_nbbo(self, symbol: str, all_bids: List, all_asks: List):
        """Calculate National Best Bid and Offer"""
        
        nbbo = self.consolidated[symbol]['nbbo']
        
        # Find best bid (highest price)
        if all_bids:
            best_bid = all_bids[0]
            
            # Aggregate size at NBBO
            nbbo_bid_size = sum(
                bid['size'] for bid in all_bids
                if bid['price'] == best_bid['price']
            )
            
            # Get all exchanges at NBBO
            nbbo_exchanges = [
                bid['exchange'] for bid in all_bids
                if bid['price'] == best_bid['price']
            ]
            
            nbbo['bid'] = {
                'price': best_bid['price'],
                'size': nbbo_bid_size,
                'exchange': nbbo_exchanges[0],  # Primary exchange
                'all_exchanges': nbbo_exchanges
            }
            
        # Find best ask (lowest price)
        if all_asks:
            best_ask = all_asks[0]
            
            # Aggregate size at NBBO
            nbbo_ask_size = sum(
                ask['size'] for ask in all_asks
                if ask['price'] == best_ask['price']
            )
            
            # Get all exchanges at NBBO
            nbbo_exchanges = [
                ask['exchange'] for ask in all_asks
                if ask['price'] == best_ask['price']
            ]
            
            nbbo['ask'] = {
                'price': best_ask['price'],
                'size': nbbo_ask_size,
                'exchange': nbbo_exchanges[0],  # Primary exchange
                'all_exchanges': nbbo_exchanges
            }
            
        # Calculate spread
        if nbbo['bid']['price'] > 0 and nbbo['ask']['price'] < float('inf'):
            nbbo['spread'] = nbbo['ask']['price'] - nbbo['bid']['price']
            nbbo['spread_bps'] = (nbbo['spread'] / nbbo['bid']['price']) * 10000
            nbbo['mid'] = (nbbo['bid']['price'] + nbbo['ask']['price']) / 2
            
    def _check_crossed_book(self, symbol: str, exchange: str):
        """Check if book is crossed on a single exchange"""
        
        book = self.books[symbol][exchange]
        
        best_bid = book.get_best('bid')
        best_ask = book.get_best('ask')
        
        if best_bid and best_ask:
            bid_price = best_bid[0]
            ask_price = best_ask[0]
            
            if bid_price >= ask_price:
                # Book is crossed!
                crossed_info = {
                    'exchange': exchange,
                    'bid': bid_price,
                    'ask': ask_price,
                    'overlap': bid_price - ask_price,
                    'time': time.time_ns()
                }
                
                self.crossed_books[symbol].append(crossed_info)
                
                # Keep only recent crossed events
                if len(self.crossed_books[symbol]) > 100:
                    self.crossed_books[symbol] = self.crossed_books[symbol][-100:]
                    
                logger.warning(f"Crossed book detected on {symbol}@{exchange}: Bid {bid_price} >= Ask {ask_price}")
                
                # Store alert in Redis
                self.redis.setex(
                    f'market:{symbol}:alert:crossed',
                    5,
                    json.dumps(crossed_info)
                )
                
    def _check_locked_crossed_nbbo(self, symbol: str):
        """Check if NBBO is locked or crossed"""
        
        nbbo = self.consolidated[symbol]['nbbo']
        
        if nbbo['bid']['price'] > 0 and nbbo['ask']['price'] < float('inf'):
            if nbbo['bid']['price'] > nbbo['ask']['price']:
                # NBBO is crossed
                logger.critical(f"CROSSED NBBO on {symbol}: Bid {nbbo['bid']['price']} > Ask {nbbo['ask']['price']}")
                
                self.redis.setex(
                    f'market:{symbol}:nbbo:crossed',
                    1,
                    'true'
                )
                
            elif nbbo['bid']['price'] == nbbo['ask']['price']:
                # NBBO is locked
                logger.warning(f"LOCKED NBBO on {symbol}: Bid == Ask == {nbbo['bid']['price']}")
                
                self.redis.setex(
                    f'market:{symbol}:nbbo:locked',
                    1,
                    'true'
                )
                
    def _calculate_fragmentation(self, symbol: str):
        """Calculate market fragmentation metrics"""
        
        # Count exchanges with quotes at NBBO
        nbbo = self.consolidated[symbol]['nbbo']
        
        if nbbo['bid']['price'] > 0:
            num_at_bid = len(nbbo['bid'].get('all_exchanges', []))
        else:
            num_at_bid = 0
            
        if nbbo['ask']['price'] < float('inf'):
            num_at_ask = len(nbbo['ask'].get('all_exchanges', []))
        else:
            num_at_ask = 0
            
        # Calculate Herfindahl index for concentration
        total_bid_size = 0
        total_ask_size = 0
        exchange_shares = defaultdict(float)
        
        for exchange, book in self.books[symbol].items():
            bid_depth = book.get_depth('bid', 1)
            ask_depth = book.get_depth('ask', 1)
            
            if bid_depth:
                size = bid_depth[0]['size']
                total_bid_size += size
                exchange_shares[exchange] += size
                
            if ask_depth:
                size = ask_depth[0]['size']
                total_ask_size += size
                exchange_shares[exchange] += size
                
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = 0
        total_size = total_bid_size + total_ask_size
        
        if total_size > 0:
            for exchange, size in exchange_shares.items():
                market_share = size / total_size
                hhi += market_share ** 2
                
        # Store fragmentation metrics
        fragmentation = {
            'exchanges_at_bid': num_at_bid,
            'exchanges_at_ask': num_at_ask,
            'hhi': hhi,  # Lower = more fragmented, Higher = more concentrated
            'effective_exchanges': 1 / hhi if hhi > 0 else 1,  # Effective number of exchanges
            'timestamp': time.time_ns()
        }
        
        self.fragmentation[symbol] = fragmentation
        
        # Store in Redis
        self.redis.setex(
            f'market:{symbol}:fragmentation',
            1,
            orjson.dumps(fragmentation).decode('utf-8')
        )
        
    def _store_exchange_books(self, symbol: str):
        """Store all exchange books and consolidated data in Redis"""
        
        pipe = self.redis.pipeline()
        
        # Store each exchange's book
        for exchange, book in self.books[symbol].items():
            book_data = {
                'bids': book.get_depth('bid', 10),
                'asks': book.get_depth('ask', 10),
                'last_update': book.last_update,
                'sequence': book.sequence,
                'checksum': book.checksum
            }
            
            pipe.setex(
                f'market:{symbol}:book:{exchange}',
                1,
                orjson.dumps(book_data).decode('utf-8')
            )
            
        # Store consolidated book
        pipe.setex(
            f'market:{symbol}:book:consolidated',
            1,
            orjson.dumps(self.consolidated[symbol]).decode('utf-8')
        )
        
        # Store NBBO separately for quick access
        nbbo = self.consolidated[symbol]['nbbo']
        pipe.setex(
            f'market:{symbol}:nbbo',
            1,
            orjson.dumps(nbbo).decode('utf-8')
        )
        
        # Store routing statistics
        pipe.setex(
            f'market:{symbol}:routing:stats',
            60,
            orjson.dumps(dict(self.routing_stats[symbol])).decode('utf-8')
        )
        
        pipe.execute()
        
    def _request_snapshot(self, symbol: str, exchange: str):
        """Request snapshot to resync book (would trigger IBKR snapshot request)"""
        
        logger.info(f"Requesting snapshot for {symbol} on {exchange}")
        
        # Set flag for main ingestion to request snapshot
        self.redis.setex(
            f'market:{symbol}:snapshot:needed',
            10,
            exchange
        )
        
    def get_nbbo(self, symbol: str) -> Dict:
        """Get current NBBO for symbol"""
        
        return self.consolidated[symbol]['nbbo']
        
    def get_exchange_book(self, symbol: str, exchange: str) -> Optional[ExchangeBook]:
        """Get book for specific exchange"""
        
        if symbol in self.books and exchange in self.books[symbol]:
            return self.books[symbol][exchange]
            
        return None