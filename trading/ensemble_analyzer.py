import pandas as pd
import numpy as np
import talib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleAnalyzer:
    """5-Model Ensemble Trading System"""
    
    def __init__(self):
        self.models = {
            'sma_crossover': self._sma_crossover,
            'rsi': self._rsi_analysis,
            'macd': self._macd_analysis,
            'bollinger_bands': self._bollinger_bands,
            'price_momentum': self._price_momentum
        }
    
    def analyze(self, data):
        """
        Analyze forex data using all 5 models
        
        Args:
            data (pandas.DataFrame): OHLCV data
        
        Returns:
            dict: Analysis results with signals and probabilities
        """
        try:
            if data is None or data.empty or len(data) < 50:
                return self._default_analysis()
            
            # Ensure we have enough data
            data = data.tail(100)  # Use last 100 periods
            
            votes = {}
            details = {}
            
            # Run each model
            for model_name, model_func in self.models.items():
                try:
                    vote, detail = model_func(data)
                    votes[model_name] = vote
                    details[model_name] = detail
                except Exception as e:
                    logger.warning(f"Error in {model_name}: {str(e)}")
                    votes[model_name] = 0
                    details[model_name] = {'error': str(e)}
            
            # Calculate ensemble results
            buy_votes = sum(1 for vote in votes.values() if vote == 1)
            sell_votes = sum(1 for vote in votes.values() if vote == -1)
            neutral_votes = sum(1 for vote in votes.values() if vote == 0)
            
            total_votes = len(votes)
            buy_probability = (buy_votes / total_votes) * 100
            sell_probability = (sell_votes / total_votes) * 100
            
            # Determine final signal (60% threshold)
            if buy_probability >= 60:
                final_signal = 'BUY'
            elif sell_probability >= 60:
                final_signal = 'SELL'
            else:
                final_signal = 'HOLD'
            
            return {
                'final_signal': final_signal,
                'buy_probability': round(buy_probability, 1),
                'sell_probability': round(sell_probability, 1),
                'model_votes': votes,
                'model_details': details,
                'vote_counts': {
                    'buy': buy_votes,
                    'sell': sell_votes,
                    'neutral': neutral_votes
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble analysis: {str(e)}")
            return self._default_analysis()
    
    def _sma_crossover(self, data):
        """Moving Average Crossover Strategy (SMA15 vs SMA40)"""
        try:
            close_prices = data['Close'].values
            sma_15 = talib.SMA(close_prices, timeperiod=15)
            sma_40 = talib.SMA(close_prices, timeperiod=40)
            
            current_15 = sma_15[-1]
            current_40 = sma_40[-1]
            prev_15 = sma_15[-2]
            prev_40 = sma_40[-2]
            
            # Check for crossover
            if prev_15 <= prev_40 and current_15 > current_40:
                return 1, {'signal': 'Golden Cross', 'sma_15': current_15, 'sma_40': current_40}
            elif prev_15 >= prev_40 and current_15 < current_40:
                return -1, {'signal': 'Death Cross', 'sma_15': current_15, 'sma_40': current_40}
            else:
                return 0, {'signal': 'No Cross', 'sma_15': current_15, 'sma_40': current_40}
                
        except Exception as e:
            return 0, {'error': str(e)}
    
    def _rsi_analysis(self, data):
        """RSI Analysis with 35/65 levels"""
        try:
            close_prices = data['Close'].values
            rsi = talib.RSI(close_prices, timeperiod=14)
            current_rsi = rsi[-1]
            
            if current_rsi < 35:
                return 1, {'rsi': current_rsi, 'signal': 'Oversold'}
            elif current_rsi > 65:
                return -1, {'rsi': current_rsi, 'signal': 'Overbought'}
            else:
                return 0, {'rsi': current_rsi, 'signal': 'Neutral'}
                
        except Exception as e:
            return 0, {'error': str(e)}
    
    def _macd_analysis(self, data):
        """MACD Analysis (12,26,9) with histogram"""
        try:
            close_prices = data['Close'].values
            macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            
            current_macd = macd[-1]
            current_signal = macd_signal[-1]
            current_hist = macd_hist[-1]
            prev_hist = macd_hist[-2]
            
            # Check for MACD crossover and histogram direction
            if current_macd > current_signal and current_hist > prev_hist:
                return 1, {'macd': current_macd, 'signal': current_signal, 'histogram': current_hist, 'signal': 'Bullish'}
            elif current_macd < current_signal and current_hist < prev_hist:
                return -1, {'macd': current_macd, 'signal': current_signal, 'histogram': current_hist, 'signal': 'Bearish'}
            else:
                return 0, {'macd': current_macd, 'signal': current_signal, 'histogram': current_hist, 'signal': 'Neutral'}
                
        except Exception as e:
            return 0, {'error': str(e)}
    
    def _bollinger_bands(self, data):
        """Bollinger Bands Analysis (20,2)"""
        try:
            close_prices = data['Close'].values
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            current_price = close_prices[-1]
            current_upper = upper[-1]
            current_lower = lower[-1]
            current_middle = middle[-1]
            
            # Calculate position within bands
            band_width = current_upper - current_lower
            price_position = (current_price - current_lower) / band_width
            
            if current_price <= current_lower:
                return 1, {'price': current_price, 'lower': current_lower, 'position': price_position, 'signal': 'Below Lower Band'}
            elif current_price >= current_upper:
                return -1, {'price': current_price, 'upper': current_upper, 'position': price_position, 'signal': 'Above Upper Band'}
            else:
                return 0, {'price': current_price, 'middle': current_middle, 'position': price_position, 'signal': 'Within Bands'}
                
        except Exception as e:
            return 0, {'error': str(e)}
    
    def _price_momentum(self, data):
        """5-period Price Momentum"""
        try:
            close_prices = data['Close'].values
            momentum = talib.MOM(close_prices, timeperiod=5)
            current_momentum = momentum[-1]
            
            # Normalize momentum as percentage
            current_price = close_prices[-1]
            momentum_pct = (current_momentum / current_price) * 100
            
            if momentum_pct > 0.1:  # Positive momentum threshold
                return 1, {'momentum': current_momentum, 'momentum_pct': momentum_pct, 'signal': 'Positive'}
            elif momentum_pct < -0.1:  # Negative momentum threshold
                return -1, {'momentum': current_momentum, 'momentum_pct': momentum_pct, 'signal': 'Negative'}
            else:
                return 0, {'momentum': current_momentum, 'momentum_pct': momentum_pct, 'signal': 'Neutral'}
                
        except Exception as e:
            return 0, {'error': str(e)}
    
    def _default_analysis(self):
        """Return default analysis when data is insufficient"""
        return {
            'final_signal': 'HOLD',
            'buy_probability': 0.0,
            'sell_probability': 0.0,
            'model_votes': {model: 0 for model in self.models.keys()},
            'model_details': {},
            'vote_counts': {'buy': 0, 'sell': 0, 'neutral': 5}
        }