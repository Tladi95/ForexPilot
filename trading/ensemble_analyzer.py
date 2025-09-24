import pandas as pd
import numpy as np
import talib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleAnalyzer:
    """Enhanced 5-Model Ensemble Trading System with Market Regime Detection"""
    
    def __init__(self):
        self.models = {
            'sma_crossover': self._sma_crossover,
            'rsi': self._rsi_analysis,
            'macd': self._macd_analysis,
            'bollinger_bands': self._bollinger_bands,
            'price_momentum': self._price_momentum
        }
        
        # Multi-timeframe weights
        self.timeframe_weights = {
            '1H': 0.30,
            '4H': 0.50,
            'Daily': 0.20
        }
    
    def analyze(self, data, multi_timeframe_data=None):
        """
        Enhanced analyze with market regime detection and multi-timeframe analysis
        
        Args:
            data (pandas.DataFrame): Primary OHLCV data (1H)
            multi_timeframe_data (dict): Optional data for 4H and Daily timeframes
        
        Returns:
            dict: Enhanced analysis results with market regime and advanced metrics
        """
        try:
            if data is None or data.empty or len(data) < 50:
                return self._default_analysis()
            
            # Ensure we have enough data
            data = data.tail(200)  # Use more data for better analysis
            
            # Market Regime Detection
            market_regime = self._detect_market_regime(data)
            
            # ATR Calculation for dynamic SL/TP
            atr_value = self._calculate_atr(data)
            
            # Price Action Analysis
            price_action = self._analyze_price_action(data)
            
            # Multi-timeframe analysis if data provided
            timeframe_consensus = self._analyze_multi_timeframe(data, multi_timeframe_data)
            
            # Original 5-model analysis
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
            
            # Apply market regime and timeframe filters
            final_signal, confidence = self._determine_enhanced_signal(
                buy_probability, sell_probability, market_regime, 
                timeframe_consensus, price_action
            )
            
            return {
                'final_signal': final_signal,
                'confidence': confidence,
                'buy_probability': round(buy_probability, 1),
                'sell_probability': round(sell_probability, 1),
                'model_votes': votes,
                'model_details': details,
                'vote_counts': {
                    'buy': buy_votes,
                    'sell': sell_votes,
                    'neutral': neutral_votes
                },
                'market_regime': market_regime,
                'atr_value': atr_value,
                'price_action': price_action,
                'timeframe_consensus': timeframe_consensus
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced ensemble analysis: {str(e)}")
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
    
    def _detect_market_regime(self, data):
        """Detect market regime using ADX and Bollinger Band Width"""
        try:
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            # ADX calculation for trend strength
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            current_adx = adx[-1]
            
            # Bollinger Band Width for volatility
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            band_width = (upper[-1] - lower[-1]) / middle[-1] * 100
            
            # Classify market regime
            if current_adx > 25:
                regime = 'TRENDING'
            elif band_width < 2.0:  # Low volatility threshold
                regime = 'RANGING'
            else:
                regime = 'VOLATILE'
            
            return {
                'regime': regime,
                'adx': round(current_adx, 1),
                'band_width': round(band_width, 2),
                'trend_strength': 'Strong' if current_adx > 40 else 'Moderate' if current_adx > 25 else 'Weak'
            }
            
        except Exception as e:
            logger.warning(f"Error in market regime detection: {str(e)}")
            return {'regime': 'UNKNOWN', 'adx': 0, 'band_width': 0, 'trend_strength': 'Unknown'}
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range for dynamic SL/TP"""
        try:
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
            current_atr = atr[-1]
            
            # Calculate ATR-based levels
            current_price = close_prices[-1]
            
            return {
                'value': round(current_atr, 5),
                'sl_multiplier': 1.5,
                'tp_multiplier': 3.0,
                'buy_sl': round(current_price - (current_atr * 1.5), 5),
                'buy_tp': round(current_price + (current_atr * 3.0), 5),
                'sell_sl': round(current_price + (current_atr * 1.5), 5),
                'sell_tp': round(current_price - (current_atr * 3.0), 5)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating ATR: {str(e)}")
            return {'value': 0, 'sl_multiplier': 1.5, 'tp_multiplier': 3.0}
    
    def _analyze_price_action(self, data):
        """Analyze price action for support/resistance and patterns"""
        try:
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            # Support and resistance levels (20-period high/low)
            resistance = np.max(high_prices[-20:])
            support = np.min(low_prices[-20:])
            current_price = close_prices[-1]
            
            # Distance to key levels
            distance_to_resistance = abs(current_price - resistance) / current_price * 100
            distance_to_support = abs(current_price - support) / current_price * 100
            
            # Check if near key levels (within 0.2%)
            near_resistance = distance_to_resistance < 0.2
            near_support = distance_to_support < 0.2
            
            # Doji pattern detection (simple version)
            open_prices = data['Open'].values
            last_open = open_prices[-1]
            last_close = close_prices[-1]
            last_high = high_prices[-1]
            last_low = low_prices[-1]
            
            body_size = abs(last_close - last_open)
            candle_range = last_high - last_low
            
            # Doji if body is less than 10% of total range
            is_doji = (body_size / candle_range) < 0.1 if candle_range > 0 else False
            
            return {
                'support': round(support, 5),
                'resistance': round(resistance, 5),
                'distance_to_support': round(distance_to_support, 2),
                'distance_to_resistance': round(distance_to_resistance, 2),
                'near_key_level': bool(near_resistance or near_support),
                'doji_pattern': bool(is_doji),
                'price_position': 'Near Resistance' if near_resistance else 'Near Support' if near_support else 'Clear'
            }
            
        except Exception as e:
            logger.warning(f"Error in price action analysis: {str(e)}")
            return {
                'support': 0, 'resistance': 0, 'near_key_level': False, 
                'doji_pattern': False, 'price_position': 'Unknown'
            }
    
    def _analyze_multi_timeframe(self, data_1h, multi_timeframe_data=None):
        """Analyze multiple timeframes for consensus"""
        try:
            if not multi_timeframe_data:
                # If no multi-timeframe data, just analyze 1H
                return {'consensus': 'NEUTRAL', 'agreement': 33, 'timeframe_signals': {'1H': 'NEUTRAL'}}
            
            timeframe_signals = {}
            
            # Analyze each timeframe
            for tf, tf_data in multi_timeframe_data.items():
                if tf_data is not None and not tf_data.empty and len(tf_data) >= 50:
                    # Run simplified analysis on each timeframe
                    votes = {}
                    for model_name, model_func in self.models.items():
                        try:
                            vote, _ = model_func(tf_data.tail(100))
                            votes[model_name] = vote
                        except:
                            votes[model_name] = 0
                    
                    buy_votes = sum(1 for vote in votes.values() if vote == 1)
                    sell_votes = sum(1 for vote in votes.values() if vote == -1)
                    
                    if buy_votes >= 3:
                        timeframe_signals[tf] = 'BUY'
                    elif sell_votes >= 3:
                        timeframe_signals[tf] = 'SELL'
                    else:
                        timeframe_signals[tf] = 'NEUTRAL'
                else:
                    timeframe_signals[tf] = 'NEUTRAL'
            
            # Calculate weighted consensus
            weighted_score = 0
            for tf, signal in timeframe_signals.items():
                weight = self.timeframe_weights.get(tf, 0)
                if signal == 'BUY':
                    weighted_score += weight
                elif signal == 'SELL':
                    weighted_score -= weight
            
            # Determine consensus
            if weighted_score > 0.3:
                consensus = 'BUY'
            elif weighted_score < -0.3:
                consensus = 'SELL'
            else:
                consensus = 'NEUTRAL'
            
            # Calculate agreement percentage
            buy_count = sum(1 for signal in timeframe_signals.values() if signal == 'BUY')
            sell_count = sum(1 for signal in timeframe_signals.values() if signal == 'SELL')
            total_timeframes = len(timeframe_signals)
            
            agreement = max(buy_count, sell_count) / total_timeframes * 100
            
            return {
                'consensus': consensus,
                'agreement': round(agreement, 0),
                'weighted_score': round(weighted_score, 2),
                'timeframe_signals': timeframe_signals
            }
            
        except Exception as e:
            logger.warning(f"Error in multi-timeframe analysis: {str(e)}")
            return {'consensus': 'NEUTRAL', 'agreement': 0, 'timeframe_signals': {}}
    
    def _determine_enhanced_signal(self, buy_prob, sell_prob, market_regime, timeframe_consensus, price_action):
        """Determine final signal with enhanced filtering"""
        try:
            # Base signal from original ensemble
            if buy_prob >= 60:
                base_signal = 'BUY'
            elif sell_prob >= 60:
                base_signal = 'SELL'
            else:
                base_signal = 'HOLD'
            
            # Start with base confidence
            confidence = max(buy_prob, sell_prob)
            
            # Apply market regime filter
            if market_regime['regime'] == 'TRENDING' and market_regime['adx'] > 30:
                confidence += 10  # Boost confidence in strong trends
            elif market_regime['regime'] == 'RANGING':
                confidence -= 15  # Reduce confidence in ranging markets
            
            # Apply timeframe consensus filter
            if timeframe_consensus['consensus'] == base_signal:
                confidence += 15  # Boost for timeframe agreement
            elif timeframe_consensus['consensus'] != 'NEUTRAL' and timeframe_consensus['consensus'] != base_signal:
                confidence -= 20  # Reduce for timeframe disagreement
                base_signal = 'HOLD'  # Override signal if strong disagreement
            
            # Apply price action filter
            if price_action['near_key_level']:
                confidence -= 10  # Reduce confidence near key levels
                if confidence < 50:  # If too risky, hold
                    base_signal = 'HOLD'
            
            if price_action['doji_pattern']:
                confidence -= 5  # Slight reduction for indecision patterns
            
            # Cap confidence at 100%
            confidence = min(100, max(0, confidence))
            
            # Final signal determination
            if confidence < 50:
                final_signal = 'HOLD'
            else:
                final_signal = base_signal
            
            # Add confirmation level
            if confidence >= 80:
                status = 'CONFIRMED'
            elif confidence >= 65:
                status = 'LIKELY'
            elif confidence >= 50:
                status = 'POSSIBLE'
            else:
                status = 'WEAK'
            
            return final_signal, {
                'percentage': round(confidence, 1),
                'status': status,
                'factors': {
                    'base_ensemble': f"{max(buy_prob, sell_prob)}%",
                    'market_regime': market_regime['regime'],
                    'timeframe_agreement': f"{timeframe_consensus['agreement']}%",
                    'price_action': price_action['price_position']
                }
            }
            
        except Exception as e:
            logger.warning(f"Error in enhanced signal determination: {str(e)}")
            return 'HOLD', {'percentage': 0, 'status': 'ERROR', 'factors': {}}
    
    def _default_analysis(self):
        """Return default analysis when data is insufficient"""
        return {
            'final_signal': 'HOLD',
            'confidence': {'percentage': 0, 'status': 'INSUFFICIENT_DATA'},
            'buy_probability': 0.0,
            'sell_probability': 0.0,
            'model_votes': {model: 0 for model in self.models.keys()},
            'model_details': {},
            'vote_counts': {'buy': 0, 'sell': 0, 'neutral': 5},
            'market_regime': {'regime': 'UNKNOWN', 'adx': 0, 'band_width': 0, 'trend_strength': 'Unknown'},
            'atr_value': {'value': 0, 'sl_multiplier': 1.5, 'tp_multiplier': 3.0},
            'price_action': {'support': 0, 'resistance': 0, 'near_key_level': False, 'doji_pattern': False, 'price_position': 'Unknown'},
            'timeframe_consensus': {'consensus': 'NEUTRAL', 'agreement': 0, 'timeframe_signals': {}}
        }