"""
ML Model - Implementation Plan Week 2 Day 3-4
Trained on Alpha Vantage historical data (20 years available!)
"""
import xgboost as xgb
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import pandas as pd
from pathlib import Path

from src.core.config import config
from src.core.logger import get_logger
from src.analytics.features import feature_engine


logger = get_logger(__name__)


class MLPredictor:
    """
    ML model for predictions using Alpha Vantage data
    Reused for paper and live trading
    Implementation Plan Week 2 Day 3-4
    """
    
    def __init__(self):
        self.features = feature_engine
        self.model = None
        self.scaler = StandardScaler()
        self.confidence_threshold = config.ml['model'].get('confidence_threshold', 0.6)
        
        # Model path
        self.model_path = Path(config.ml['model']['path'])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def load_model(self):
        """Load trained model or create default"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                scaler_path = self.model_path.with_suffix('.scaler.pkl')
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded existing model from {self.model_path}")
            else:
                self._create_default_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_default_model()
    
    def _create_default_model(self):
        """Create default XGBoost model"""
        logger.info("Creating new XGBoost model with defaults")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=4,  # BUY_CALL, BUY_PUT, HOLD, CLOSE
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    
    async def train_with_av_historical(self, symbols: List[str], 
                                      days_back: int = 30):
        """
        Train model using Alpha Vantage historical options data
        AV provides 20 years of history with Greeks!
        """
        logger.info(f"Training with {days_back} days of Alpha Vantage historical data...")
        
        X = []
        y = []
        
        for symbol in symbols:
            try:
                # Get historical options data from Alpha Vantage
                hist_options = await self.features.options.get_historical_options_ml_data(
                    symbol, days_back
                )
                
                # Get historical price data for labels
                hist_prices = await self.features.options.market.get_bars(
                    symbol, f'{days_back} D'
                )
                
                if hist_options.empty or hist_prices.empty:
                    logger.warning(f"No historical data for {symbol}")
                    continue
                
                # Generate training samples
                for i in range(min(len(hist_options), len(hist_prices)) - 12):
                    # Calculate features using Alpha Vantage data
                    features = await self.features.calculate_features(symbol)
                    X.append(features)
                    
                    # Generate label based on price movement
                    if i + 12 < len(hist_prices):
                        future_return = (hist_prices.iloc[i+12]['close'] - 
                                       hist_prices.iloc[i]['close']) / hist_prices.iloc[i]['close']
                        
                        if future_return > 0.002:  # 0.2% up
                            y.append(0)  # BUY_CALL
                        elif future_return < -0.002:  # 0.2% down
                            y.append(1)  # BUY_PUT
                        else:
                            y.append(2)  # HOLD
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X, y)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.model_path.with_suffix('.scaler.pkl'))
            
            logger.info(f"Model trained on {len(X)} samples from Alpha Vantage historical data")
        else:
            logger.warning("No training data available")
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Make prediction - USED BY SIGNAL GENERATOR
        Returns: (signal, confidence)
        """
        if self.model is None:
            return 'HOLD', 0.0
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction probabilities
            probs = self.model.predict_proba(features_scaled)[0]
            
            # Get best prediction
            prediction = np.argmax(probs)
            confidence = probs[prediction]
            
            # Map to signal
            signals = ['BUY_CALL', 'BUY_PUT', 'HOLD', 'CLOSE']
            signal = signals[prediction]
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                signal = 'HOLD'
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 'HOLD', 0.0


# BUILD ON FEATURE ENGINE
ml_model = MLPredictor()
