import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self):
        # Find and load .env file
        config_dir = Path(__file__).parent.parent.parent / 'config'
        env_path = config_dir / '.env'
        load_dotenv(env_path)
        
        # Load environment variables
        self.av_api_key = os.getenv('AV_API_KEY')
        self.database_url = os.getenv('DATABASE_URL')
        
        # Load Alpha Vantage YAML config
        av_config_path = config_dir / 'apis' / 'alpha_vantage.yaml'
        if av_config_path.exists():
            with open(av_config_path, 'r') as f:
                self.av_config = yaml.safe_load(f) or {}
        else:
            self.av_config = {}
    
    def get(self, key, default=None):
        """Simple getter for config values"""
        return getattr(self, key, default)