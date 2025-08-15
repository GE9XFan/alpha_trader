#!/usr/bin/env python3
"""
Configuration Structure Generator
Creates all YAML configuration templates
Version: 1.0
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigGenerator:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.config_dir = self.base_path / "config"
        
    def create_all_configs(self):
        """Create all configuration files"""
        print("\n🔧 Creating Configuration Structure...")
        print("=" * 60)
        
        # System configurations
        self.create_system_configs()
        
        # API configurations
        self.create_api_configs()
        
        # Data management configurations
        self.create_data_configs()
        
        # Strategy configurations
        self.create_strategy_configs()
        
        # Risk configurations
        self.create_risk_configs()
        
        # ML configurations
        self.create_ml_configs()
        
        # Execution configurations
        self.create_execution_configs()
        
        # Monitoring configurations
        self.create_monitoring_configs()
        
        # Environment configurations
        self.create_environment_configs()
        
        print("\n" + "=" * 60)
        print("✅ Configuration structure complete!")
    
    def save_yaml(self, filepath: Path, data: Dict[str, Any]):
        """Save dictionary as YAML file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Created: {filepath.relative_to(self.base_path)}")
    
    def create_system_configs(self):
        """Create system configuration files"""
        print("\n📁 Creating system configurations...")
        
        # Database configuration
        db_config = {
            'database': {
                'host': '${DB_HOST}',  # From environment variable
                'port': 5432,
                'name': '${DB_NAME}',
                'user': '${DB_USER}',
                'password': '${DB_PASSWORD}',
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'echo': False,  # Set True for SQL debugging
                'connection_timeout': 10,
                'command_timeout': 60,
                'options': {
                    'sslmode': 'prefer',
                    'connect_timeout': 10
                }
            },
            'migrations': {
                'auto_upgrade': False,  # Manual migrations in production
                'backup_before_migration': True
            }
        }
        self.save_yaml(self.config_dir / 'system' / 'database.yaml', db_config)
        
        # Redis configuration
        redis_config = {
            'redis': {
                'host': '${REDIS_HOST}',
                'port': 6379,
                'password': '${REDIS_PASSWORD}',
                'db': 0,
                'decode_responses': True,
                'max_connections': 50,
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'socket_keepalive': True,
                'health_check_interval': 30,
                'retry_on_timeout': True
            },
            'cache': {
                'default_ttl': 300,  # 5 minutes
                'max_ttl': 86400,    # 24 hours
                'key_prefix': 'trading:',
                'serializer': 'json'
            }
        }
        self.save_yaml(self.config_dir / 'system' / 'redis.yaml', redis_config)
        
        # Logging configuration
        logging_config = {
            'logging': {
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    },
                    'detailed': {
                        'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
                    }
                },
                'handlers': {
                    'console': {
                        'class': 'logging.StreamHandler',
                        'level': 'INFO',
                        'formatter': 'standard',
                        'stream': 'ext://sys.stdout'
                    },
                    'file': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': 'DEBUG',
                        'formatter': 'detailed',
                        'filename': 'logs/trading_system.log',
                        'maxBytes': 10485760,  # 10MB
                        'backupCount': 5
                    },
                    'error_file': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': 'ERROR',
                        'formatter': 'detailed',
                        'filename': 'logs/errors.log',
                        'maxBytes': 10485760,
                        'backupCount': 5
                    }
                },
                'loggers': {
                    'src': {
                        'level': 'DEBUG',
                        'handlers': ['console', 'file'],
                        'propagate': False
                    },
                    'trading': {
                        'level': 'INFO',
                        'handlers': ['console', 'file', 'error_file'],
                        'propagate': False
                    }
                },
                'root': {
                    'level': 'INFO',
                    'handlers': ['console', 'file']
                }
            }
        }
        self.save_yaml(self.config_dir / 'system' / 'logging.yaml', logging_config)
        
        # Paths configuration
        paths_config = {
            'paths': {
                'data_dir': './data',
                'cache_dir': './data/cache',
                'raw_data_dir': './data/raw',
                'processed_data_dir': './data/processed',
                'models_dir': './models',
                'logs_dir': './logs',
                'reports_dir': './reports',
                'backups_dir': './backups'
            }
        }
        self.save_yaml(self.config_dir / 'system' / 'paths.yaml', paths_config)
    
    def create_api_configs(self):
        """Create API configuration files"""
        print("\n📡 Creating API configurations...")
        
        # Alpha Vantage configuration
        av_config = {
            'alpha_vantage': {
                'api_key': '${AV_API_KEY}',
                'base_url': 'https://www.alphavantage.co/query',
                'timeout': 30,
                'max_retries': 3,
                'retry_delay': 1,
                'retry_backoff': 2,
                'endpoints': {
                    # Options APIs
                    'REALTIME_OPTIONS': {
                        'cache_ttl': 10,
                        'priority': 1
                    },
                    'HISTORICAL_OPTIONS': {
                        'cache_ttl': 86400,
                        'priority': 3
                    },
                    # Core Indicators
                    'RSI': {
                        'cache_ttl': 60,
                        'priority': 1,
                        'default_params': {
                            'interval': '5min',
                            'time_period': 14,
                            'series_type': 'close'
                        }
                    },
                    'MACD': {
                        'cache_ttl': 60,
                        'priority': 1,
                        'default_params': {
                            'interval': '5min',
                            'series_type': 'close',
                            'fastperiod': 12,
                            'slowperiod': 26,
                            'signalperiod': 9
                        }
                    },
                    'BBANDS': {
                        'cache_ttl': 60,
                        'priority': 1,
                        'default_params': {
                            'interval': '5min',
                            'time_period': 20,
                            'series_type': 'close',
                            'nbdevup': 2,
                            'nbdevdn': 2
                        }
                    },
                    'VWAP': {
                        'cache_ttl': 60,
                        'priority': 1,
                        'default_params': {
                            'interval': '5min'
                        }
                    },
                    'ATR': {
                        'cache_ttl': 300,
                        'priority': 2,
                        'default_params': {
                            'interval': '5min',
                            'time_period': 14
                        }
                    },
                    'ADX': {
                        'cache_ttl': 300,
                        'priority': 2,
                        'default_params': {
                            'interval': '5min',
                            'time_period': 14
                        }
                    }
                    # Add remaining 37 APIs as needed
                }
            }
        }
        self.save_yaml(self.config_dir / 'apis' / 'alpha_vantage.yaml', av_config)
        
        # IBKR configuration
        ibkr_config = {
            'ibkr': {
                'username': '${IBKR_USERNAME}',
                'password': '${IBKR_PASSWORD}',
                'account': '${IBKR_ACCOUNT}',
                'trading_mode': 'paper',  # CRITICAL: Start with paper
                'gateway_host': '127.0.0.1',
                'gateway_port': 4001,
                'client_id': 1,
                'connection_timeout': 30,
                'readonly': False,
                'reconnect_attempts': 3,
                'reconnect_delay': 5,
                'max_concurrent_subscriptions': 50,
                'data_feeds': {
                    'quotes': {
                        'enabled': True,
                        'fields': ['bid', 'ask', 'last', 'bid_size', 'ask_size', 'volume']
                    },
                    'bars': {
                        'enabled': True,
                        'sizes': ['1 secs', '5 secs', '1 min', '5 mins', '15 mins', '30 mins', '1 hour']
                    },
                    'moc_imbalance': {
                        'enabled': True,
                        'start_time': '15:40',
                        'end_time': '15:55'
                    }
                }
            }
        }
        self.save_yaml(self.config_dir / 'apis' / 'ibkr.yaml', ibkr_config)
        
        # Rate limits configuration
        rate_limits_config = {
            'rate_limits': {
                'alpha_vantage': {
                    'calls_per_minute': 600,  # Hard limit
                    'target_calls_per_minute': 500,  # Target to stay safe
                    'tokens_per_second': 10,
                    'burst_size': 20,
                    'warning_threshold': 450,
                    'backoff_multiplier': 2,
                    'max_backoff': 60
                },
                'ibkr': {
                    'max_subscriptions': 50,
                    'messages_per_second': 50,
                    'orders_per_second': 5
                }
            }
        }
        self.save_yaml(self.config_dir / 'apis' / 'rate_limits.yaml', rate_limits_config)
    
    def create_data_configs(self):
        """Create data management configuration files"""
        print("\n📊 Creating data configurations...")
        
        # Symbols configuration
        symbols_config = {
            'symbols': {
                'tier_a': {
                    'symbols': ['SPY', 'QQQ', 'IWM', 'SPX'],
                    'priority': 1,
                    'update_frequency': {
                        'options': 12,  # seconds
                        'indicators': 60,
                        'analytics': 300
                    }
                },
                'tier_b': {
                    'symbols': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA'],
                    'priority': 2,
                    'update_frequency': {
                        'options': 45,
                        'indicators': 300,
                        'analytics': 900
                    }
                },
                'tier_c': {
                    'symbols': [],  # Dynamic watchlist
                    'max_symbols': 20,
                    'priority': 3,
                    'update_frequency': {
                        'options': 180,
                        'indicators': 600,
                        'analytics': 1800
                    }
                }
            },
            'symbol_filters': {
                'min_volume': 1000000,
                'min_price': 5.0,
                'max_price': 10000.0,
                'min_options_volume': 100
            }
        }
        self.save_yaml(self.config_dir / 'data' / 'symbols.yaml', symbols_config)
        
        # Schedules configuration
        schedules_config = {
            'schedules': {
                'market_hours': {
                    'pre_market_start': '04:00',
                    'pre_market_end': '09:30',
                    'market_open': '09:30',
                    'market_close': '16:00',
                    'after_hours_end': '20:00',
                    'timezone': 'America/New_York'
                },
                'data_collection': {
                    'tier_a': {
                        'options_with_greeks': {
                            'interval_seconds': 12,
                            'priority': 1
                        },
                        'rsi': {
                            'interval_seconds': 60,
                            'priority': 1
                        },
                        'macd': {
                            'interval_seconds': 60,
                            'priority': 1
                        },
                        'bbands': {
                            'interval_seconds': 60,
                            'priority': 1
                        },
                        'vwap': {
                            'interval_seconds': 60,
                            'priority': 1
                        },
                        'atr': {
                            'interval_seconds': 300,
                            'priority': 2
                        },
                        'adx': {
                            'interval_seconds': 300,
                            'priority': 2
                        }
                    },
                    'tier_b': {
                        'options_with_greeks': {
                            'interval_seconds': 45,
                            'priority': 2
                        },
                        'indicators_bundle': {
                            'interval_seconds': 300,
                            'priority': 2
                        }
                    },
                    'tier_c': {
                        'options_with_greeks': {
                            'interval_seconds': 180,
                            'priority': 3
                        },
                        'indicators_bundle': {
                            'interval_seconds': 600,
                            'priority': 3
                        }
                    }
                },
                'moc_window': {
                    'enabled': True,
                    'start_time': '15:40',
                    'end_time': '15:55',
                    'priority_boost': 10,
                    'update_interval': 5
                }
            }
        }
        self.save_yaml(self.config_dir / 'data' / 'schedules.yaml', schedules_config)
        
        # Ingestion configuration
        ingestion_config = {
            'ingestion': {
                'batch_size': 1000,
                'commit_interval': 100,
                'error_threshold': 0.05,  # 5% error rate triggers alert
                'retry_failed': True,
                'max_retries': 3,
                'dead_letter_queue': True,
                'validation': {
                    'enabled': True,
                    'strict_mode': False,  # True in production
                    'log_validation_errors': True
                }
            }
        }
        self.save_yaml(self.config_dir / 'data' / 'ingestion.yaml', ingestion_config)
        
        # Validation configuration
        validation_config = {
            'validation': {
                'price_data': {
                    'max_price_change_pct': 20,  # 20% max change
                    'min_price': 0.01,
                    'max_price': 100000,
                    'required_fields': ['open', 'high', 'low', 'close', 'volume']
                },
                'greeks': {
                    'delta': {
                        'min': -1.0,
                        'max': 1.0
                    },
                    'gamma': {
                        'min': 0.0,
                        'max': 1.0
                    },
                    'theta': {
                        'calls_max': 0.0,
                        'puts_min': 0.0
                    },
                    'vega': {
                        'min': 0.0,
                        'max': None
                    },
                    'max_age_seconds': 30,
                    'required_fields': ['delta', 'gamma', 'theta', 'vega']
                },
                'indicators': {
                    'rsi': {
                        'min': 0,
                        'max': 100
                    },
                    'max_age_seconds': 300
                }
            }
        }
        self.save_yaml(self.config_dir / 'data' / 'validation.yaml', validation_config)
    
    def create_strategy_configs(self):
        """Create strategy configuration files"""
        print("\n🎯 Creating strategy configurations...")
        
        # 0DTE Strategy
        zero_dte_config = {
            'strategy': {
                'name': 'Zero DTE Strategy',
                'enabled': True,
                'description': 'Trades options expiring same day'
            },
            'confidence': {
                'minimum': 0.75,
                'ml_weight': 0.40,
                'indicators_weight': 0.30,
                'greeks_weight': 0.30
            },
            'timing': {
                'entry_window': {
                    'start': '09:45',
                    'end': '14:00'
                },
                'auto_close': '15:30',
                'no_entry_after': '14:00'
            },
            'position_limits': {
                'max_concurrent': 3,
                'max_per_symbol': 1,
                'min_premium': 0.50,
                'max_premium': 10.00,
                'max_contracts': 10
            },
            'entry_rules': {
                'rsi': {
                    'enabled': True,
                    'min_value': 30,
                    'max_value': 70,
                    'weight': 0.15
                },
                'delta': {
                    'enabled': True,
                    'min_abs_value': 0.25,
                    'max_abs_value': 0.75,
                    'weight': 0.20
                },
                'gamma': {
                    'enabled': True,
                    'max_value': 0.20,
                    'weight': 0.15
                },
                'theta_decay': {
                    'enabled': True,
                    'min_ratio': 0.03,  # theta/price
                    'weight': 0.15
                },
                'implied_volatility': {
                    'enabled': True,
                    'min_percentile': 20,
                    'weight': 0.10
                },
                'volume': {
                    'enabled': True,
                    'min_ratio': 0.5,  # current/average
                    'weight': 0.10
                },
                'bid_ask_spread': {
                    'enabled': True,
                    'max_spread': 0.10,
                    'weight': 0.15
                }
            },
            'exit_rules': {
                'stop_loss': 0.25,  # 25% loss
                'take_profit': 0.50,  # 50% gain
                'time_stop': '15:30',
                'trailing_stop': {
                    'enabled': True,
                    'activation': 0.20,  # Activate at 20% profit
                    'distance': 0.10     # Trail by 10%
                }
            }
        }
        self.save_yaml(self.config_dir / 'strategies' / '0dte.yaml', zero_dte_config)
        
        # 1DTE Strategy
        one_dte_config = {
            'strategy': {
                'name': 'One DTE Strategy',
                'enabled': True,
                'description': 'Trades options expiring next day'
            },
            'confidence': {
                'minimum': 0.70,
                'ml_weight': 0.35,
                'indicators_weight': 0.35,
                'greeks_weight': 0.30
            },
            'timing': {
                'entry_window': {
                    'start': '09:45',
                    'end': '15:00'
                },
                'hold_overnight': True
            },
            'position_limits': {
                'max_concurrent': 5,
                'max_per_symbol': 2,
                'min_premium': 0.30,
                'max_premium': 15.00,
                'max_contracts': 15
            },
            'entry_rules': {
                'rsi': {
                    'enabled': True,
                    'min_value': 25,
                    'max_value': 75,
                    'weight': 0.15
                },
                'delta': {
                    'enabled': True,
                    'min_abs_value': 0.20,
                    'max_abs_value': 0.80,
                    'weight': 0.20
                },
                'gamma': {
                    'enabled': True,
                    'max_value': 0.25,
                    'weight': 0.15
                },
                'theta_decay': {
                    'enabled': True,
                    'min_ratio': 0.02,
                    'weight': 0.15
                }
            },
            'exit_rules': {
                'stop_loss': 0.30,
                'take_profit': 0.60,
                'overnight_hedge': True
            }
        }
        self.save_yaml(self.config_dir / 'strategies' / '1dte.yaml', one_dte_config)
        
        # Swing 14D Strategy
        swing_config = {
            'strategy': {
                'name': '14-Day Swing Strategy',
                'enabled': True,
                'description': 'Longer-term options trades (1-14 days)'
            },
            'confidence': {
                'minimum': 0.65,
                'ml_weight': 0.30,
                'indicators_weight': 0.40,
                'greeks_weight': 0.30
            },
            'timing': {
                'entry_window': {
                    'start': '09:30',
                    'end': '16:00'
                },
                'min_hold_days': 1,
                'max_hold_days': 14
            },
            'position_limits': {
                'max_concurrent': 10,
                'max_per_symbol': 3,
                'min_premium': 1.00,
                'max_premium': 50.00,
                'max_contracts': 20
            },
            'entry_rules': {
                'trend_alignment': {
                    'enabled': True,
                    'weight': 0.25
                },
                'support_resistance': {
                    'enabled': True,
                    'weight': 0.20
                },
                'volume_analysis': {
                    'enabled': True,
                    'weight': 0.15
                }
            },
            'exit_rules': {
                'stop_loss': 0.20,
                'take_profit': 0.80,
                'time_decay_exit': True,
                'roll_conditions': {
                    'enabled': True,
                    'min_profit': 0.30,
                    'days_before_expiry': 2
                }
            }
        }
        self.save_yaml(self.config_dir / 'strategies' / 'swing_14d.yaml', swing_config)
        
        # MOC Imbalance Strategy
        moc_config = {
            'strategy': {
                'name': 'MOC Imbalance Strategy',
                'enabled': True,
                'description': 'Trades based on market-on-close imbalances'
            },
            'confidence': {
                'minimum': 0.70,
                'imbalance_weight': 0.50,
                'technical_weight': 0.30,
                'ml_weight': 0.20
            },
            'timing': {
                'active_window': {
                    'start': '15:40',
                    'end': '15:55'
                },
                'decision_deadline': '15:50',
                'execution_window': {
                    'start': '15:50',
                    'end': '15:55'
                }
            },
            'thresholds': {
                'min_imbalance': 10000000,  # $10M
                'min_normalized_imbalance': 0.10,  # 10% of avg volume
                'min_iv': 0.15,
                'max_spread': 0.05
            },
            'execution': {
                'use_straddle': {
                    'when_iv_below': 0.20,
                    'max_premium': 5.00
                },
                'use_directional': {
                    'when_iv_above': 0.20,
                    'follow_imbalance': True
                }
            }
        }
        self.save_yaml(self.config_dir / 'strategies' / 'moc_imbalance.yaml', moc_config)
    
    def create_risk_configs(self):
        """Create risk management configuration files"""
        print("\n⚠️ Creating risk configurations...")
        
        # Position limits
        position_limits_config = {
            'position_limits': {
                'greeks': {
                    'max_delta': 0.80,
                    'min_delta': 0.20,
                    'max_gamma': 0.20,
                    'max_vega': 200,
                    'min_theta_ratio': 0.02
                },
                'sizing': {
                    'max_position_size': 0.05,  # 5% of capital
                    'max_contract_value': 1000,
                    'min_contract_value': 50,
                    'max_contracts_per_trade': 20
                },
                'entry_checks': [
                    'delta_within_range',
                    'gamma_below_limit',
                    'theta_decay_sufficient',
                    'spread_acceptable',
                    'volume_sufficient',
                    'capital_available'
                ]
            }
        }
        self.save_yaml(self.config_dir / 'risk' / 'position_limits.yaml', position_limits_config)
        
        # Portfolio limits
        portfolio_limits_config = {
            'portfolio_limits': {
                'greeks': {
                    'max_net_delta': 0.30,
                    'max_net_gamma': 0.75,
                    'max_net_vega': 1000,
                    'max_net_theta': -500
                },
                'exposure': {
                    'max_capital_at_risk': 0.20,  # 20% of total
                    'max_sector_concentration': 0.30,
                    'max_symbol_concentration': 0.15,
                    'max_strategy_concentration': 0.40
                },
                'daily_limits': {
                    'max_trades': 20,
                    'max_loss': 0.02,  # 2% daily loss limit
                    'max_consecutive_losses': 3,
                    'max_new_positions': 10
                }
            }
        }
        self.save_yaml(self.config_dir / 'risk' / 'portfolio_limits.yaml', portfolio_limits_config)
        
        # Circuit breakers
        circuit_breakers_config = {
            'circuit_breakers': {
                'daily_loss': {
                    'enabled': True,
                    'threshold': 0.02,  # 2%
                    'action': 'halt_new_trades',
                    'notification': 'immediate'
                },
                'weekly_loss': {
                    'enabled': True,
                    'threshold': 0.05,  # 5%
                    'action': 'reduce_position_sizes',
                    'notification': 'immediate'
                },
                'drawdown': {
                    'enabled': True,
                    'threshold': 0.10,  # 10%
                    'action': 'emergency_shutdown',
                    'notification': 'immediate'
                },
                'rapid_losses': {
                    'enabled': True,
                    'losses_count': 5,
                    'time_window_minutes': 60,
                    'action': 'pause_trading',
                    'pause_duration_minutes': 30
                },
                'vix_spike': {
                    'enabled': True,
                    'threshold': 40,
                    'action': 'close_all_positions',
                    'notification': 'immediate'
                }
            }
        }
        self.save_yaml(self.config_dir / 'risk' / 'circuit_breakers.yaml', circuit_breakers_config)
        
        # Position sizing
        sizing_config = {
            'sizing': {
                'method': 'kelly_criterion',  # or 'fixed', 'volatility_adjusted'
                'kelly': {
                    'fraction': 0.25,  # Use 25% of Kelly
                    'max_allocation': 0.05,
                    'min_allocation': 0.01
                },
                'volatility_adjustment': {
                    'enabled': True,
                    'base_size': 0.02,
                    'vol_scalar': 0.5
                },
                'account_allocation': {
                    'max_per_strategy': {
                        '0dte': 0.30,
                        '1dte': 0.30,
                        'swing_14d': 0.30,
                        'moc_imbalance': 0.10
                    }
                }
            }
        }
        self.save_yaml(self.config_dir / 'risk' / 'sizing.yaml', sizing_config)
    
    def create_ml_configs(self):
        """Create ML configuration files"""
        print("\n🤖 Creating ML configurations...")
        
        # Models configuration
        models_config = {
            'models': {
                'paths': {
                    'zero_dte_model': 'models/zero_dte_model.pkl',
                    'one_dte_model': 'models/one_dte_model.pkl',
                    'swing_model': 'models/swing_model.pkl',
                    'moc_model': 'models/moc_model.pkl'
                },
                'ensemble': {
                    'enabled': True,
                    'voting': 'soft',  # or 'hard'
                    'weights': [0.3, 0.3, 0.2, 0.2]
                },
                'update_schedule': {
                    'enabled': False,  # Models are frozen for MVP
                    'frequency': 'weekly'
                }
            }
        }
        self.save_yaml(self.config_dir / 'ml' / 'models.yaml', models_config)
        
        # Features configuration
        features_config = {
            'features': {
                'price_features': [
                    'returns_1m', 'returns_5m', 'returns_15m',
                    'volatility_5m', 'volatility_15m',
                    'high_low_ratio', 'close_open_ratio'
                ],
                'volume_features': [
                    'volume_ratio', 'volume_momentum',
                    'buy_sell_imbalance'
                ],
                'greeks_features': [
                    'delta', 'gamma', 'theta', 'vega',
                    'delta_change', 'gamma_change',
                    'theta_decay_rate'
                ],
                'indicator_features': [
                    'rsi', 'macd_signal', 'macd_histogram',
                    'bb_position', 'bb_width',
                    'atr', 'adx', 'vwap_distance'
                ],
                'market_features': [
                    'spy_correlation', 'sector_momentum',
                    'vix_level', 'market_breadth'
                ],
                'scaling': {
                    'method': 'standard',  # or 'minmax', 'robust'
                    'clip_outliers': True,
                    'outlier_threshold': 3.0
                }
            }
        }
        self.save_yaml(self.config_dir / 'ml' / 'features.yaml', features_config)
        
        # Thresholds configuration
        thresholds_config = {
            'thresholds': {
                'confidence': {
                    'minimum': 0.60,
                    'high_confidence': 0.80,
                    'very_high_confidence': 0.90
                },
                'prediction': {
                    'binary_threshold': 0.50,
                    'multi_class_threshold': 0.35
                },
                'feature_importance': {
                    'minimum': 0.01,
                    'log_important': 0.05
                }
            }
        }
        self.save_yaml(self.config_dir / 'ml' / 'thresholds.yaml', thresholds_config)
    
    def create_execution_configs(self):
        """Create execution configuration files"""
        print("\n⚡ Creating execution configurations...")
        
        # Trading hours configuration
        trading_hours_config = {
            'trading_hours': {
                'regular_session': {
                    'start': '09:30',
                    'end': '16:00',
                    'timezone': 'America/New_York'
                },
                'extended_hours': {
                    'pre_market': {
                        'enabled': False,
                        'start': '04:00',
                        'end': '09:30'
                    },
                    'after_hours': {
                        'enabled': False,
                        'start': '16:00',
                        'end': '20:00'
                    }
                },
                'no_trade_windows': [
                    {
                        'name': 'First 15 minutes',
                        'start': '09:30',
                        'end': '09:45',
                        'strategies_affected': ['0dte', '1dte']
                    },
                    {
                        'name': 'Last 30 minutes (except MOC)',
                        'start': '15:30',
                        'end': '16:00',
                        'strategies_affected': ['0dte'],
                        'exceptions': ['moc_imbalance']
                    }
                ],
                'holidays': {
                    'check_enabled': True,
                    'source': 'NYSE',
                    'early_close_days': ['Christmas Eve', 'Thanksgiving Friday']
                }
            }
        }
        self.save_yaml(self.config_dir / 'execution' / 'trading_hours.yaml', trading_hours_config)
        
        # Order types configuration
        order_types_config = {
            'order_types': {
                'default': 'LIMIT',
                'allowed_types': ['LIMIT', 'MARKET', 'STOP', 'STOP_LIMIT'],
                'entry_orders': {
                    'type': 'LIMIT',
                    'time_in_force': 'DAY',
                    'price_adjustment': 0.02,  # 2 cents from mid
                    'max_chase_ticks': 3,
                    'cancel_after_seconds': 30
                },
                'exit_orders': {
                    'take_profit': {
                        'type': 'LIMIT',
                        'time_in_force': 'GTC'
                    },
                    'stop_loss': {
                        'type': 'STOP',
                        'time_in_force': 'GTC',
                        'trail_amount': None
                    },
                    'emergency_exit': {
                        'type': 'MARKET',
                        'time_in_force': 'IOC'
                    }
                },
                'smart_routing': {
                    'enabled': True,
                    'destination': 'SMART'
                }
            }
        }
        self.save_yaml(self.config_dir / 'execution' / 'order_types.yaml', order_types_config)
        
        # Slippage configuration
        slippage_config = {
            'slippage': {
                'model': 'linear',  # or 'square_root', 'logarithmic'
                'base_slippage_bps': 5,  # 5 basis points
                'factors': {
                    'spread': 0.5,  # 50% of spread
                    'volatility': 0.1,
                    'size': 0.05,
                    'time_of_day': 0.02
                },
                'max_acceptable_slippage_bps': 20,
                'tracking': {
                    'enabled': True,
                    'alert_threshold_bps': 15
                }
            }
        }
        self.save_yaml(self.config_dir / 'execution' / 'slippage.yaml', slippage_config)
    
    def create_monitoring_configs(self):
        """Create monitoring configuration files"""
        print("\n📈 Creating monitoring configurations...")
        
        # Alerts configuration
        alerts_config = {
            'alerts': {
                'channels': {
                    'discord': {
                        'enabled': True,
                        'priority_levels': ['INFO', 'WARNING', 'CRITICAL', 'EMERGENCY']
                    },
                    'email': {
                        'enabled': False,
                        'priority_levels': ['CRITICAL', 'EMERGENCY']
                    },
                    'sms': {
                        'enabled': False,
                        'priority_levels': ['EMERGENCY']
                    }
                },
                'types': {
                    'trade_executed': {
                        'enabled': True,
                        'level': 'INFO'
                    },
                    'stop_loss_triggered': {
                        'enabled': True,
                        'level': 'WARNING'
                    },
                    'circuit_breaker_triggered': {
                        'enabled': True,
                        'level': 'CRITICAL'
                    },
                    'system_error': {
                        'enabled': True,
                        'level': 'CRITICAL'
                    },
                    'api_rate_limit_warning': {
                        'enabled': True,
                        'level': 'WARNING',
                        'threshold': 450  # calls/minute
                    },
                    'daily_loss_limit': {
                        'enabled': True,
                        'level': 'CRITICAL'
                    }
                },
                'rate_limiting': {
                    'max_per_minute': 10,
                    'group_similar': True,
                    'group_window_seconds': 60
                }
            }
        }
        self.save_yaml(self.config_dir / 'monitoring' / 'alerts.yaml', alerts_config)
        
        # Discord configuration
        discord_config = {
            'discord': {
                'webhook_url': '${DISCORD_WEBHOOK_URL}',
                'username': 'Trading Bot',
                'avatar_url': None,
                'message_formatting': {
                    'use_embeds': True,
                    'colors': {
                        'entry': 3066993,   # Green
                        'exit': 15158332,   # Red
                        'info': 3447003,    # Blue
                        'warning': 16776960, # Yellow
                        'critical': 10038562 # Dark Red
                    }
                },
                'channels': {
                    'trades': {
                        'new_position': True,
                        'close_position': True,
                        'stop_loss': True,
                        'take_profit': True
                    },
                    'performance': {
                        'daily_summary': True,
                        'weekly_summary': True,
                        'milestone_alerts': True
                    },
                    'system': {
                        'startup': True,
                        'shutdown': True,
                        'errors': True,
                        'warnings': True
                    }
                }
            }
        }
        self.save_yaml(self.config_dir / 'monitoring' / 'discord.yaml', discord_config)
        
        # Dashboard configuration
        dashboard_config = {
            'dashboard': {
                'enabled': True,
                'host': '0.0.0.0',
                'port': 8000,
                'auto_reload': False,
                'authentication': {
                    'enabled': False,  # Enable in production
                    'type': 'basic'
                },
                'endpoints': {
                    'health': '/health',
                    'positions': '/positions',
                    'performance': '/performance',
                    'config': '/config',
                    'logs': '/logs',
                    'emergency_stop': '/emergency-stop'
                },
                'websocket': {
                    'enabled': True,
                    'port': 8001,
                    'heartbeat_interval': 30
                },
                'refresh_rates': {
                    'positions': 1,  # seconds
                    'performance': 5,
                    'system_health': 10
                }
            }
        }
        self.save_yaml(self.config_dir / 'monitoring' / 'dashboard.yaml', dashboard_config)
    
    def create_environment_configs(self):
        """Create environment-specific configuration files"""
        print("\n🌍 Creating environment configurations...")
        
        # Development environment
        dev_config = {
            'environment': {
                'name': 'development',
                'debug': True,
                'testing': True,
                'overrides': {
                    'apis': {
                        'ibkr': {
                            'trading_mode': 'paper'
                        }
                    },
                    'risk': {
                        'circuit_breakers': {
                            'daily_loss': {
                                'threshold': 0.05  # More lenient in dev
                            }
                        }
                    },
                    'logging': {
                        'level': 'DEBUG'
                    }
                }
            }
        }
        self.save_yaml(self.config_dir / 'environments' / 'development.yaml', dev_config)
        
        # Paper trading environment
        paper_config = {
            'environment': {
                'name': 'paper',
                'debug': False,
                'testing': True,
                'overrides': {
                    'apis': {
                        'ibkr': {
                            'trading_mode': 'paper'
                        }
                    },
                    'risk': {
                        'circuit_breakers': {
                            'daily_loss': {
                                'threshold': 0.02  # Production-like
                            }
                        }
                    },
                    'logging': {
                        'level': 'INFO'
                    },
                    'monitoring': {
                        'alerts': {
                            'channels': {
                                'discord': {
                                    'enabled': True
                                }
                            }
                        }
                    }
                }
            }
        }
        self.save_yaml(self.config_dir / 'environments' / 'paper.yaml', paper_config)
        
        # Production environment
        prod_config = {
            'environment': {
                'name': 'production',
                'debug': False,
                'testing': False,
                'overrides': {
                    'apis': {
                        'ibkr': {
                            'trading_mode': 'live'  # REAL MONEY
                        }
                    },
                    'risk': {
                        'circuit_breakers': {
                            'daily_loss': {
                                'threshold': 0.02,
                                'action': 'emergency_shutdown'
                            }
                        }
                    },
                    'logging': {
                        'level': 'WARNING'
                    },
                    'data': {
                        'validation': {
                            'strict_mode': True
                        }
                    },
                    'monitoring': {
                        'alerts': {
                            'channels': {
                                'discord': {
                                    'enabled': True
                                },
                                'email': {
                                    'enabled': True
                                }
                            }
                        }
                    }
                }
            }
        }
        self.save_yaml(self.config_dir / 'environments' / 'production.yaml', prod_config)


def main():
    """Generate all configuration files"""
    generator = ConfigGenerator()
    generator.create_all_configs()
    
    print("\n📋 Configuration Summary:")
    print("  • System configs: database, redis, logging, paths")
    print("  • API configs: Alpha Vantage, IBKR, rate limits")
    print("  • Data configs: symbols, schedules, validation")
    print("  • Strategy configs: 0DTE, 1DTE, Swing, MOC")
    print("  • Risk configs: position/portfolio limits, circuit breakers")
    print("  • ML configs: models, features, thresholds")
    print("  • Execution configs: trading hours, order types")
    print("  • Monitoring configs: alerts, Discord, dashboard")
    print("  • Environment configs: dev, paper, production")
    
    print("\n⚠️ Important Notes:")
    print("  1. All values with ${} are loaded from environment variables")
    print("  2. IBKR defaults to 'paper' mode - change only when ready for production")
    print("  3. Review and adjust all limits before going live")
    print("  4. Test configuration loading before proceeding")


if __name__ == "__main__":
    main()