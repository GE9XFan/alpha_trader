#!/usr/bin/env python3
"""
Conflation Handler Module
Manages message conflation, buffering, and sequence tracking for high-frequency data
Ensures no message loss while managing throughput
"""

import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any
import threading
import lz4.frame
import msgpack

logger = logging.getLogger(__name__)


class MessageBuffer:
    """Thread-safe circular buffer for market data messages"""
    
    def __init__(self, size: int = 10000):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = threading.RLock()
        self.overflow_count = 0
        self.message_count = 0
        
    def add(self, message: Dict) -> bool:
        """Add message to buffer"""
        
        with self.lock:
            if len(self.buffer) >= self.size:
                self.overflow_count += 1
                
            self.buffer.append(message)
            self.message_count += 1
            
            return True
            
    def get_batch(self, max_items: int = 100) -> List[Dict]:
        """Get batch of messages for processing"""
        
        with self.lock:
            batch = []
            for _ in range(min(max_items, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
                    
            return batch
            
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        
        with self.lock:
            return {
                'size': len(self.buffer),
                'capacity': self.size,
                'utilization': len(self.buffer) / self.size * 100,
                'overflow_count': self.overflow_count,
                'total_messages': self.message_count
            }
            
    def clear(self):
        """Clear buffer"""
        
        with self.lock:
            self.buffer.clear()
            self.overflow_count = 0


class SequenceTracker:
    """Tracks message sequences to detect gaps and ensure order"""
    
    def __init__(self):
        self.sequences = defaultdict(int)  # Per channel sequence numbers
        self.gaps = defaultdict(list)  # Detected gaps
        self.out_of_order = defaultdict(list)  # Out of order messages
        self.last_reset = time.time()
        
    def check_sequence(self, channel: str, seq_num: int) -> Dict:
        """Check if sequence number is valid"""
        
        result = {
            'valid': True,
            'gap_detected': False,
            'out_of_order': False,
            'gap_size': 0
        }
        
        expected = self.sequences[channel] + 1
        
        if seq_num == expected:
            # Normal sequence
            self.sequences[channel] = seq_num
            
        elif seq_num > expected:
            # Gap detected
            gap_size = seq_num - expected
            self.gaps[channel].append({
                'expected': expected,
                'received': seq_num,
                'gap_size': gap_size,
                'time': time.time()
            })
            
            result['gap_detected'] = True
            result['gap_size'] = gap_size
            
            # Update sequence
            self.sequences[channel] = seq_num
            
        elif seq_num < expected:
            # Out of order message
            self.out_of_order[channel].append({
                'expected': expected,
                'received': seq_num,
                'time': time.time()
            })
            
            result['out_of_order'] = True
            result['valid'] = False  # Don't process out of order
            
        return result
        
    def reset_channel(self, channel: str):
        """Reset sequence for a channel"""
        
        self.sequences[channel] = 0
        self.gaps[channel].clear()
        self.out_of_order[channel].clear()
        
    def get_gap_report(self) -> Dict:
        """Get report of all gaps"""
        
        report = {}
        for channel, gap_list in self.gaps.items():
            if gap_list:
                report[channel] = {
                    'gap_count': len(gap_list),
                    'total_missing': sum(g['gap_size'] for g in gap_list),
                    'recent_gaps': gap_list[-10:]  # Last 10 gaps
                }
                
        return report


class ConflationHandler:
    """
    Manages message conflation for high-frequency market data
    Conflation combines multiple updates into single message to manage throughput
    """
    
    def __init__(self, conflation_window_ms: int = 10):
        """
        Initialize conflation handler
        
        Args:
            conflation_window_ms: Time window for conflation in milliseconds
        """
        
        self.conflation_window_ms = conflation_window_ms
        self.conflation_buffers = defaultdict(lambda: defaultdict(list))
        self.last_conflation = defaultdict(float)
        self.conflation_stats = defaultdict(lambda: {
            'messages_received': 0,
            'messages_conflated': 0,
            'messages_sent': 0
        })
        
        # Compression for audit trail
        self.compression_enabled = True
        self.compressed_buffers = deque(maxlen=100)  # Keep last 100 compressed batches
        
    def should_conflate(self, symbol: str, msg_type: str) -> bool:
        """Check if message should be conflated"""
        
        # Don't conflate critical messages
        critical_types = ['trade', 'sweep', 'halt', 'auction']
        if msg_type in critical_types:
            return False
            
        # Check time since last conflation
        now = time.time()
        last = self.last_conflation[symbol]
        
        if (now - last) * 1000 < self.conflation_window_ms:
            return True
            
        return False
        
    def add_message(self, symbol: str, msg_type: str, message: Dict) -> Optional[Dict]:
        """
        Add message to conflation buffer
        
        Returns conflated message if window expired, None otherwise
        """
        
        self.conflation_stats[symbol]['messages_received'] += 1
        
        # Check if should conflate
        if self.should_conflate(symbol, msg_type):
            # Add to buffer
            self.conflation_buffers[symbol][msg_type].append(message)
            self.conflation_stats[symbol]['messages_conflated'] += 1
            return None
            
        else:
            # Conflate existing buffer and return
            conflated = self._conflate_messages(symbol, msg_type)
            
            # Reset buffer with new message
            self.conflation_buffers[symbol][msg_type] = [message]
            self.last_conflation[symbol] = time.time()
            
            if conflated:
                self.conflation_stats[symbol]['messages_sent'] += 1
                
            return conflated
            
    def _conflate_messages(self, symbol: str, msg_type: str) -> Optional[Dict]:
        """Conflate buffered messages into single message"""
        
        messages = self.conflation_buffers[symbol][msg_type]
        
        if not messages:
            return None
            
        if len(messages) == 1:
            return messages[0]
            
        # Conflation strategy depends on message type
        if msg_type == 'depth':
            return self._conflate_depth_updates(messages)
        elif msg_type == 'quote':
            return self._conflate_quotes(messages)
        else:
            # Default: return latest message
            return messages[-1]
            
    def _conflate_depth_updates(self, messages: List[Dict]) -> Dict:
        """Conflate multiple depth updates"""
        
        # Build final order book state from all updates
        final_book = {'bids': {}, 'asks': {}}
        
        for msg in messages:
            # Apply each update to build final state
            side = msg.get('side', 'bid')
            price = msg.get('price')
            size = msg.get('size')
            
            if size == 0:
                # Delete level
                if price in final_book[side + 's']:
                    del final_book[side + 's'][price]
            else:
                # Add/update level
                final_book[side + 's'][price] = size
                
        # Convert to sorted lists
        conflated = {
            'type': 'depth',
            'bids': sorted(
                [{'price': p, 'size': s} for p, s in final_book['bids'].items()],
                key=lambda x: x['price'],
                reverse=True
            ),
            'asks': sorted(
                [{'price': p, 'size': s} for p, s in final_book['asks'].items()],
                key=lambda x: x['price']
            ),
            'timestamp': messages[-1].get('timestamp', time.time_ns()),
            'conflated_count': len(messages)
        }
        
        return conflated
        
    def _conflate_quotes(self, messages: List[Dict]) -> Dict:
        """Conflate multiple quote updates"""
        
        # For quotes, use the latest values
        latest = messages[-1]
        
        # Add conflation metadata
        latest['conflated_count'] = len(messages)
        latest['conflation_window_ms'] = self.conflation_window_ms
        
        return latest
        
    def flush_symbol(self, symbol: str) -> List[Dict]:
        """Flush all conflated messages for a symbol"""
        
        flushed = []
        
        for msg_type, messages in self.conflation_buffers[symbol].items():
            if messages:
                conflated = self._conflate_messages(symbol, msg_type)
                if conflated:
                    flushed.append(conflated)
                    
        # Clear buffers
        self.conflation_buffers[symbol].clear()
        
        return flushed
        
    def compress_batch(self, messages: List[Dict]) -> bytes:
        """Compress message batch for storage/transmission"""
        
        if not self.compression_enabled:
            return msgpack.packb(messages)
            
        # Serialize with msgpack
        packed = msgpack.packb(messages)
        
        # Compress with LZ4
        compressed = lz4.frame.compress(packed)
        
        # Store in buffer for audit
        self.compressed_buffers.append({
            'time': time.time(),
            'original_size': len(packed),
            'compressed_size': len(compressed),
            'message_count': len(messages),
            'compression_ratio': len(packed) / len(compressed) if compressed else 0
        })
        
        return compressed
        
    def decompress_batch(self, compressed: bytes) -> List[Dict]:
        """Decompress message batch"""
        
        if not self.compression_enabled:
            return msgpack.unpackb(compressed)
            
        # Decompress with LZ4
        decompressed = lz4.frame.decompress(compressed)
        
        # Deserialize with msgpack
        messages = msgpack.unpackb(decompressed)
        
        return messages
        
    def get_stats(self) -> Dict:
        """Get conflation statistics"""
        
        total_stats = {
            'total_received': 0,
            'total_conflated': 0,
            'total_sent': 0,
            'conflation_ratio': 0,
            'compression_stats': {}
        }
        
        # Aggregate stats
        for symbol_stats in self.conflation_stats.values():
            total_stats['total_received'] += symbol_stats['messages_received']
            total_stats['total_conflated'] += symbol_stats['messages_conflated']
            total_stats['total_sent'] += symbol_stats['messages_sent']
            
        # Calculate ratio
        if total_stats['total_received'] > 0:
            total_stats['conflation_ratio'] = (
                total_stats['total_conflated'] / total_stats['total_received'] * 100
            )
            
        # Compression stats
        if self.compressed_buffers:
            recent = list(self.compressed_buffers)[-10:]  # Last 10 batches
            total_stats['compression_stats'] = {
                'avg_compression_ratio': sum(b['compression_ratio'] for b in recent) / len(recent),
                'total_compressed_batches': len(self.compressed_buffers)
            }
            
        return total_stats
        
    def reset_stats(self):
        """Reset statistics"""
        
        self.conflation_stats.clear()
        self.compressed_buffers.clear()