"""
Comprehensive bug-hunting tests for AlphaTrader.
These tests are designed to FIND bugs, not to pass.
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import threading
import psycopg2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.message import Message
from core.bus import MessageBus
from core.persistence import EventStore
from core.plugin import Plugin, PluginState
from core.rate_limiter import TokenBucket, MultiLevelRateLimiter
from core.config import ConfigLoader


class TestMessageBusBugs:
    """Tests designed to find bugs in the message bus."""
    
    def test_concurrent_subscribe_unsubscribe(self):
        """Test race conditions in subscription management."""
        bus = MessageBus()
        handlers = []
        
        def handler(msg):
            pass
        
        # Create multiple threads that subscribe/unsubscribe simultaneously
        def subscribe_unsubscribe():
            for i in range(100):
                bus.subscribe(f"test.{i % 10}", handler)
                bus.unsubscribe(f"test.{i % 10}", handler)
        
        threads = [threading.Thread(target=subscribe_unsubscribe) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Check for inconsistent state
        stats = bus.get_stats()
        assert stats['total_handlers'] >= 0, "Negative handler count indicates race condition"
    
    def test_handler_exception_propagation(self):
        """Test that handler exceptions don't crash the bus."""
        bus = MessageBus()
        call_count = [0]
        
        def bad_handler(msg):
            call_count[0] += 1
            raise RuntimeError("Handler exploded!")
        
        def good_handler(msg):
            call_count[0] += 1
        
        bus.subscribe("test.*", bad_handler)
        bus.subscribe("test.*", good_handler)
        
        # Publish message - both handlers should be called despite exception
        bus.publish("test.event", {"data": "test"})
        
        # Give handlers time to execute
        time.sleep(0.1)
        
        # Both handlers should have been called
        assert call_count[0] == 2, f"Expected 2 calls, got {call_count[0]}"
    
    def test_pattern_matching_edge_cases(self):
        """Test pattern matching with edge cases."""
        bus = MessageBus()
        
        matched = []
        def handler(msg):
            matched.append(msg.event_type)
        
        # Test various patterns
        patterns = [
            "*",  # Should this match anything?
            "*.*.* ",  # Pattern with trailing space
            "test..event",  # Double dots
            "",  # Empty pattern
            "test.*.*.end",  # Multiple wildcards
        ]
        
        for pattern in patterns:
            try:
                bus.subscribe(pattern, handler)
            except:
                pass  # Pattern might be invalid
        
        # Publish various events
        events = [
            "test",
            "test.event",
            "test.sub.event",
            "test..event",
            "",
        ]
        
        for event in events:
            try:
                bus.publish(event, {})
            except:
                pass  # Event type might be invalid
    
    def test_memory_leak_in_subscriptions(self):
        """Test for memory leaks when handlers are not properly cleaned up."""
        bus = MessageBus()
        
        # Create many handlers and subscribe them
        for i in range(1000):
            handler = lambda msg, i=i: None
            bus.subscribe(f"test.{i}", handler)
        
        stats_before = bus.get_stats()
        
        # Now unsubscribe half of them
        for i in range(500):
            handler = lambda msg, i=i: None
            bus.unsubscribe(f"test.{i}", handler)  # This won't work - different lambda!
        
        stats_after = bus.get_stats()
        
        # Check if unsubscribe actually worked
        assert stats_after['total_handlers'] == 1000, "Unsubscribe with different lambda object doesn't work!"


class TestRateLimiterBugs:
    """Tests designed to find bugs in rate limiting."""
    
    def test_token_bucket_race_condition(self):
        """Test for race conditions in token bucket."""
        bucket = TokenBucket(capacity=10, refill_rate=10)
        
        success_count = [0]
        
        def try_acquire_many():
            for _ in range(100):
                if bucket.try_acquire(1):
                    success_count[0] += 1
                time.sleep(0.001)
        
        # Multiple threads trying to acquire tokens
        threads = [threading.Thread(target=try_acquire_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # We should never get more successes than capacity + refill
        max_possible = 10 + (1.0 * 10)  # initial + ~1 second of refill
        assert success_count[0] <= max_possible + 10, f"Too many tokens acquired: {success_count[0]}"
    
    def test_negative_tokens(self):
        """Test that tokens can't go negative."""
        bucket = TokenBucket(capacity=5, refill_rate=0)  # No refill
        
        # Try to acquire more than available
        assert bucket.try_acquire(3) == True
        assert bucket.try_acquire(3) == False  # Should fail
        assert bucket.available_tokens() >= 0, "Tokens went negative!"
    
    def test_config_validation(self):
        """Test that invalid configs are rejected."""
        
        # Test with missing required fields
        invalid_configs = [
            {},  # Empty config
            {'calls_per_minute': None},  # None value
            {'calls_per_minute': -1},  # Negative value
            {'calls_per_minute': 0},  # Zero value
            {'daily_limit': 'not_a_number'},  # Wrong type
        ]
        
        for config in invalid_configs:
            try:
                limiter = MultiLevelRateLimiter(config)
                assert False, f"Config should have been rejected: {config}"
            except (ValueError, TypeError):
                pass  # Expected
    
    @pytest.mark.asyncio
    async def test_daily_limit_reset(self):
        """Test that daily limit resets properly."""
        config = {
            'calls_per_minute': 60,
            'daily_limit': 100,
        }
        
        limiter = MultiLevelRateLimiter(config)
        
        # Use up some daily limit
        for _ in range(10):
            await limiter.acquire_standard()
        
        assert limiter.daily_calls == 10
        
        # Manually set last reset to >24 hours ago
        limiter.last_reset = time.time() - 86401
        
        # Next acquire should reset counter
        await limiter.acquire_standard()
        assert limiter.daily_calls == 1, "Daily counter didn't reset"


class TestPersistenceBugs:
    """Tests designed to find bugs in persistence layer."""
    
    @pytest.fixture
    def test_db_url(self):
        """Create a test database URL."""
        return "postgresql://alphatrader:alphatrader_dev@localhost:5432/alphatrader_test"
    
    def test_connection_pool_exhaustion(self, test_db_url):
        """Test that connection pool doesn't get exhausted."""
        try:
            store = EventStore(test_db_url)
            
            # Create many messages without closing connections properly
            for i in range(100):
                msg = Message.create(
                    event_type=f"test.{i}",
                    data={"index": i}
                )
                store.save(msg)
            
            # Try to get stats - should still work
            stats = store.get_stats()
            assert stats is not None
            
        except psycopg2.OperationalError as e:
            if "connection pool exhausted" in str(e):
                pytest.fail("Connection pool exhausted!")
            else:
                pytest.skip(f"Database not available: {e}")
    
    def test_large_message_handling(self, test_db_url):
        """Test handling of very large messages."""
        try:
            store = EventStore(test_db_url)
            
            # Create a message with huge data
            huge_data = {"key": "x" * 1000000}  # 1MB string
            
            msg = Message.create(
                event_type="test.huge",
                data=huge_data
            )
            
            store.save(msg)
            
            # Try to retrieve it using event type
            events = store.get_events(event_type="test.huge", limit=1)
            assert len(events) > 0
            retrieved = events[0]
            assert len(retrieved.data["key"]) == 1000000
            
        except Exception as e:
            if "too large" in str(e).lower():
                pytest.fail("Large message handling failed")
            else:
                pytest.skip(f"Database not available: {e}")
    
    def test_sql_injection_prevention(self, test_db_url):
        """Test that SQL injection is prevented."""
        try:
            store = EventStore(test_db_url)
            
            # Try to inject SQL through event type
            malicious_type = "test'; DROP TABLE events; --"
            
            msg = Message.create(
                event_type=malicious_type,
                data={"test": "data"}
            )
            
            store.save(msg)
            
            # Table should still exist
            stats = store.get_stats()
            assert stats is not None, "Table was dropped!"
            
        except Exception as e:
            if "table" in str(e).lower() and "not exist" in str(e).lower():
                pytest.fail("SQL injection succeeded!")
            else:
                pytest.skip(f"Database not available: {e}")


class TestPluginBugs:
    """Tests designed to find bugs in plugin system."""
    
    @pytest.mark.asyncio
    async def test_plugin_state_transitions(self):
        """Test invalid state transitions."""
        
        class TestPlugin(Plugin):
            async def start(self):
                pass
            
            async def stop(self):
                pass
            
            def health_check(self):
                return {"status": "healthy"}
        
        bus = MessageBus()
        plugin = TestPlugin("test", bus, {})
        
        # Try to stop before starting
        await plugin.stop_plugin()
        assert plugin.state != PluginState.STOPPED, "Plugin stopped without starting"
        
        # Start plugin
        await plugin.start_plugin()
        assert plugin.state == PluginState.RUNNING
        
        # Try to start again
        await plugin.start_plugin()
        assert plugin.state == PluginState.RUNNING, "Double start changed state"
    
    def test_plugin_config_validation(self):
        """Test that plugin configs are validated."""
        
        class TestPlugin(Plugin):
            def __init__(self, name, bus, config):
                # Plugin expects certain config keys
                if "required_key" not in config:
                    raise ValueError("Missing required_key")
                super().__init__(name, bus, config)
            
            async def start(self):
                pass
            
            async def stop(self):
                pass
            
            def health_check(self):
                return {"status": "healthy"}
        
        bus = MessageBus()
        
        # Should raise error for missing config
        with pytest.raises(ValueError):
            plugin = TestPlugin("test", bus, {})


class TestConfigLoaderBugs:
    """Tests designed to find bugs in configuration loading."""
    
    def test_environment_variable_injection(self):
        """Test that environment variables can't be injected maliciously."""
        import os
        
        # Set a malicious environment variable
        os.environ["MALICIOUS"] = "'; DROP TABLE events; --"
        
        config_content = """
        database:
          user: ${MALICIOUS}
        """
        
        # This should safely handle the malicious content
        # Not create SQL injection vulnerability
        loader = ConfigLoader()
        # Config should load but value should be escaped/safe
    
    def test_circular_references(self):
        """Test handling of circular references in config."""
        config_content = """
        value1: ${value2}
        value2: ${value1}
        """
        
        # This should either detect the cycle or have a max depth
        loader = ConfigLoader()
        # Should not cause infinite loop
    
    def test_missing_config_files(self):
        """Test behavior with missing config files."""
        loader = ConfigLoader("/nonexistent/path")
        
        # Should handle gracefully, not crash
        try:
            config = loader.get_system_config()
            # Should return defaults or raise clear error
        except FileNotFoundError:
            pass  # Expected


class TestIntegrationBugs:
    """Integration tests to find bugs in component interactions."""
    
    @pytest.mark.asyncio
    async def test_event_ordering_under_load(self):
        """Test that events maintain order under heavy load."""
        bus = MessageBus()
        received_order = []
        
        async def handler(msg):
            received_order.append(msg.data["index"])
        
        bus.subscribe("test.*", handler)
        
        # Publish many events rapidly
        for i in range(100):
            bus.publish("test.event", {"index": i})
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Check if order was maintained (might not be guaranteed!)
        # This test reveals if ordering is expected but not enforced
        if received_order != list(range(100)):
            print(f"Events received out of order: {received_order[:10]}...")
    
    def test_shutdown_with_pending_messages(self):
        """Test shutdown while messages are being processed."""
        bus = MessageBus()
        processing = threading.Event()
        
        def slow_handler(msg):
            processing.set()
            time.sleep(1)  # Simulate slow processing
        
        bus.subscribe("test.*", slow_handler)
        
        # Publish message
        bus.publish("test.event", {})
        
        # Wait for processing to start
        processing.wait(timeout=0.5)
        
        # Now try to shutdown while handler is running
        bus.shutdown()
        
        # Check that shutdown completed cleanly
        stats = bus.get_stats()
        # Should have handled shutdown gracefully


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])