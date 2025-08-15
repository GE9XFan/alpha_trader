# Quick test - save as test_config_load.py
import yaml

with open('config/strategies/0dte.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print("✓ 0DTE Config loaded successfully!")
    print(f"  Min confidence: {config['confidence']['minimum']}")
    print(f"  Entry window: {config['timing']['entry_window']['start']} - {config['timing']['entry_window']['end']}")
    print(f"  Max concurrent positions: {config['position_limits']['max_concurrent']}")