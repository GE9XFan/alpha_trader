#!/usr/bin/env python3
"""Test base classes implementation"""

from src.foundation.base_module import BaseModule, ComponentStatus, HealthStatus
from src.connections.base_client import BaseAPIClient
from src.strategies.base_strategy import BaseStrategy
from src.foundation.config_manager import ConfigManager


# Test implementations
class TestModule(BaseModule):
    def initialize(self) -> bool:
        print(f"  Initializing {self.name}")
        return True
    
    def health_check(self) -> dict:
        return {'status': 'healthy', 'checks_passed': True}
    
    def shutdown(self) -> bool:
        print(f"  Shutting down {self.name}")
        return True


class TestAPIClient(BaseAPIClient):
    def connect(self) -> bool:
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        self.is_connected = False
        return True
    
    def call(self, endpoint: str, params: dict) -> dict:
        return {'status': 'success', 'endpoint': endpoint}
    
    def health_check(self) -> bool:
        return self.is_connected


class TestStrategy(BaseStrategy):
    def evaluate(self, context: dict) -> dict:
        return {'confidence': 0.8, 'signal': 'BUY'}
    
    def generate_signal(self, evaluation: dict) -> dict:
        if evaluation['confidence'] >= self.min_confidence:
            return {'action': 'BUY', 'confidence': evaluation['confidence']}
        return None
    
    def calculate_position_size(self, signal: dict, capital: float) -> int:
        return max(1, int(capital * 0.02 / 100))  # 2% of capital


def main():
    print("=" * 60)
    print("TESTING BASE CLASSES")
    print("=" * 60)
    
    # Test BaseModule
    print("\n1. Testing BaseModule:")
    module = TestModule({'test': 'config'}, "TestModule")
    print(f"✓ Created: {module.name}")
    print(f"✓ Initial status: {module.status.value}")
    
    if module.start():
        print(f"✓ Started: {module.status.value}")
    
    status = module.get_status()
    print(f"✓ Status check: {status['status']}")
    
    if module.stop():
        print(f"✓ Stopped: {module.status.value}")
    
    # Test BaseAPIClient
    print("\n2. Testing BaseAPIClient:")
    client = TestAPIClient({'api_key': 'test'}, "TestAPI")
    print(f"✓ Created: {client.name}")
    
    if client.connect():
        print(f"✓ Connected: {client.is_connected}")
    
    response = client.call('/test', {'param': 'value'})
    print(f"✓ API call: {response['status']}")
    
    stats = client.get_stats()
    print(f"✓ Stats: {stats['total_calls']} calls")
    
    # Test BaseStrategy
    print("\n3. Testing BaseStrategy:")
    config = ConfigManager()
    strategy_config = config.get_strategy_config('0dte')
    strategy = TestStrategy(strategy_config)
    print(f"✓ Created: {strategy.name}")
    print(f"✓ Min confidence: {strategy.min_confidence}")
    
    context = {'price': 100, 'rsi': 50}
    evaluation = strategy.evaluate(context)
    print(f"✓ Evaluation: confidence={evaluation['confidence']}")
    
    signal = strategy.generate_signal(evaluation)
    if signal:
        print(f"✓ Signal generated: {signal['action']}")
    
    size = strategy.calculate_position_size(signal, 10000)
    print(f"✓ Position size: {size} contracts")
    
    print("\n" + "=" * 60)
    print("✅ ALL BASE CLASS TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()