import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleAnalyzer:
    """Institutional-Grade 5-Model Ensemble with Quantitative Finance Methods"""
    
    def __init__(self):
        self.models = {
            'sma_crossover': self._sma_crossover,
            'rsi': self._rsi_analysis,
            'macd': self._macd_analysis,
            'bollinger_bands': self._bollinger_bands,
            'price_momentum': self._price_momentum
        }
        
        # Intraday timeframe weights optimized for 5m-15m
        self.timeframe_weights = {
            '5m': 0.40,
            '15m': 0.35,
            '1h': 0.25
        }
        
        # Model performance tracking for Sharpe/Sortino weighting (with error handling)
        try:
            self.model_performance = self._load_model_performance()
        except Exception as e:
            logger.warning(f"Error loading model performance, using defaults: {str(e)}")
            self.model_performance = {
                model_name: {'weight': 0.2, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0}
                for model_name in self.models.keys()
            }
        
        # Probability calibrator (with safe initialization)
        self.calibrator = None
        self.meta_classifier = None
        try:
            self.scaler = StandardScaler()
        except Exception as e:
            logger.warning(f"Error initializing scaler: {str(e)}")
            self.scaler = None
        
        # Expected Value tracking
        self.ev_tracker = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_rate': 0.0
        }
        
        # Dynamic confidence thresholds based on ADX
        self.confidence_thresholds = {
            'strong_trend': {'adx_min': 40, 'threshold': 0.8},
            'moderate_trend': {'adx_min': 25, 'threshold': 0.6},
            'weak_trend': {'adx_min': 0, 'threshold': 0.4}
        }
        
        # Load or initialize calibration models with error handling
        try:
            self._load_calibration_models()
        except Exception as e:
            logger.warning(f"Error loading calibration models, using defaults: {str(e)}")
            self.calibrator = None
            self.meta_classifier = None
    
    def analyze(self, data, multi_timeframe_data=None):
        """
        Institutional-grade analysis with Sharpe weighting, probability calibration, 
        EV filtering, and meta-labeling
        
        Args:
            data (pandas.DataFrame): Primary OHLCV data (5m-15m optimized)
            multi_timeframe_data (dict): Optional multi-timeframe data
        
        Returns:
            dict: Advanced analysis with quantitative finance metrics
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
            
            # Sharpe/Sortino-weighted ensemble analysis
            votes = {}
            details = {}
            model_weights = self._calculate_model_weights(data)
            
            # Run each model with performance tracking
            for model_name, model_func in self.models.items():
                try:
                    vote, detail = model_func(data)
                    votes[model_name] = vote
                    details[model_name] = detail
                    # Update model performance for weight calculation
                    self._update_model_performance(model_name, vote, data)
                except Exception as e:
                    logger.warning(f"Error in {model_name}: {str(e)}")
                    votes[model_name] = 0
                    details[model_name] = {'error': str(e)}
            
            # Calculate Sharpe-weighted ensemble results
            weighted_signal = self._calculate_weighted_ensemble(votes, model_weights)
            raw_probability = self._calculate_raw_probability(votes, model_weights)
            
            # Probability calibration
            calibrated_probability = self._calibrate_probability(data, raw_probability)
            
            # Expected Value calculation
            ev_score = self._calculate_expected_value(calibrated_probability, market_regime)
            
            # Convert to buy/sell probabilities
            if weighted_signal > 0:
                buy_probability = calibrated_probability * 100
                sell_probability = (1 - calibrated_probability) * 100
            else:
                buy_probability = (1 - calibrated_probability) * 100
                sell_probability = calibrated_probability * 100
            
            # Dynamic confidence thresholds based on ADX
            dynamic_threshold = self._get_dynamic_threshold(market_regime)
            
            # Meta-labeling decision
            meta_decision = self._meta_labeling_decision(data, calibrated_probability, market_regime)
            
            # Apply regime-aware signal determination with EV filter
            final_signal, confidence = self._determine_institutional_signal(
                weighted_signal, calibrated_probability, market_regime, 
                timeframe_consensus, price_action, ev_score, dynamic_threshold, meta_decision
            )
            
            # Fractional Kelly position sizing
            kelly_fraction = self._calculate_kelly_fraction(calibrated_probability, ev_score)
            
            return {
                'final_signal': final_signal,
                'confidence': confidence,
                'buy_probability': round(buy_probability, 1),
                'sell_probability': round(sell_probability, 1),
                'model_votes': votes,
                'model_details': details,
                'model_weights': model_weights,
                'vote_counts': {
                    'buy': sum(1 for vote in votes.values() if vote == 1),
                    'sell': sum(1 for vote in votes.values() if vote == -1),
                    'neutral': sum(1 for vote in votes.values() if vote == 0)
                },
                'market_regime': market_regime,
                'atr_value': atr_value,
                'price_action': price_action,
                'timeframe_consensus': timeframe_consensus,
                'weighted_signal': weighted_signal,
                'calibrated_probability': calibrated_probability,
                'expected_value': ev_score,
                'kelly_fraction': kelly_fraction,
                'dynamic_threshold': dynamic_threshold,
                'meta_decision': meta_decision
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
    
    def _load_model_performance(self):
        """Load or initialize model performance tracking for Sharpe/Sortino weighting"""
        try:
            if os.path.exists('model_performance.json'):
                import json
                with open('model_performance.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading model performance: {str(e)}")
        
        # Initialize performance tracking for each model
        return {
            model_name: {
                'returns': [],
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'weight': 0.2,  # Equal weight initially
                'total_trades': 0,
                'winning_trades': 0
            } for model_name in self.models.keys()
        }
    
    def _load_calibration_models(self):
        """Load or initialize probability calibration models"""
        try:
            if os.path.exists('calibration_model.joblib'):
                self.calibrator = joblib.load('calibration_model.joblib')
            if os.path.exists('meta_classifier.joblib'):
                self.meta_classifier = joblib.load('meta_classifier.joblib')
            if os.path.exists('scaler.joblib'):
                self.scaler = joblib.load('scaler.joblib')
                
            # Load EV tracker
            if os.path.exists('ev_tracker.json'):
                import json
                with open('ev_tracker.json', 'r') as f:
                    self.ev_tracker = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading calibration models: {str(e)}")
    
    def _calculate_model_weights(self, data):
        """Calculate Sharpe/Sortino-based weights for each model"""
        try:
            weights = {}
            total_weight = 0
            
            for model_name, performance in self.model_performance.items():
                # Calculate Sharpe ratio weight
                sharpe = performance.get('sharpe_ratio', 0.0)
                sortino = performance.get('sortino_ratio', 0.0)
                
                # Combine Sharpe and Sortino with preference for Sortino
                combined_score = 0.4 * sharpe + 0.6 * sortino
                
                # Convert to positive weight (add 1 to handle negative ratios)
                weight = max(0.1, 1 + combined_score)  # Minimum weight of 0.1
                weights[model_name] = weight
                total_weight += weight
            
            # Normalize weights to sum to 1
            if total_weight > 0:
                for model_name in weights:
                    weights[model_name] /= total_weight
            else:
                # Equal weights if no performance data
                equal_weight = 1.0 / len(self.models)
                weights = {model_name: equal_weight for model_name in self.models.keys()}
            
            return weights
        except Exception as e:
            logger.warning(f"Error calculating model weights: {str(e)}")
            equal_weight = 1.0 / len(self.models)
            return {model_name: equal_weight for model_name in self.models.keys()}
    
    def _update_model_performance(self, model_name, vote, data):
        """Update model performance for Sharpe/Sortino calculation"""
        try:
            if model_name not in self.model_performance:
                return
                
            # Simulate return based on vote and actual price movement
            if len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                actual_return = (current_price - prev_price) / prev_price
                
                # Calculate model return based on vote alignment
                if vote == 1:  # Buy signal
                    model_return = actual_return
                elif vote == -1:  # Sell signal
                    model_return = -actual_return
                else:  # Hold signal
                    model_return = 0
                
                # Update returns list (keep last 100 returns for rolling calculation)
                returns = self.model_performance[model_name]['returns']
                returns.append(model_return)
                if len(returns) > 100:
                    returns.pop(0)
                
                # Calculate Sharpe and Sortino ratios
                if len(returns) >= 10:
                    returns_array = np.array(returns)
                    mean_return = np.mean(returns_array)
                    std_return = np.std(returns_array)
                    
                    # Sharpe ratio
                    if std_return > 0:
                        self.model_performance[model_name]['sharpe_ratio'] = mean_return / std_return
                    
                    # Sortino ratio (downside deviation)
                    negative_returns = returns_array[returns_array < 0]
                    if len(negative_returns) > 0:
                        downside_deviation = np.std(negative_returns)
                        if downside_deviation > 0:
                            self.model_performance[model_name]['sortino_ratio'] = mean_return / downside_deviation
                
                # Update trade counts
                self.model_performance[model_name]['total_trades'] += 1
                if model_return > 0:
                    self.model_performance[model_name]['winning_trades'] += 1
                    
        except Exception as e:
            logger.warning(f"Error updating model performance for {model_name}: {str(e)}")
    
    def _calculate_weighted_ensemble(self, votes, weights):
        """Calculate weighted ensemble signal using Sharpe/Sortino weights"""
        try:
            weighted_signal = 0
            for model_name, vote in votes.items():
                weight = weights.get(model_name, 0.2)
                weighted_signal += vote * weight
            return weighted_signal
        except Exception as e:
            logger.warning(f"Error calculating weighted ensemble: {str(e)}")
            return 0
    
    def _calculate_raw_probability(self, votes, weights):
        """Calculate raw probability before calibration"""
        try:
            weighted_signal = self._calculate_weighted_ensemble(votes, weights)
            # Convert weighted signal to probability using sigmoid-like function
            probability = 1 / (1 + np.exp(-2 * weighted_signal))
            return probability
        except Exception as e:
            logger.warning(f"Error calculating raw probability: {str(e)}")
            return 0.5
    
    def _calibrate_probability(self, data, raw_probability):
        """Calibrate probability using trained logistic regression"""
        try:
            if self.calibrator is None:
                # If no calibrator trained yet, return raw probability
                return raw_probability
            
            # Extract features for calibration
            features = self._extract_calibration_features(data)
            if features is not None:
                features_scaled = self.scaler.transform([features])
                calibrated_prob = self.calibrator.predict_proba(features_scaled)[0][1]
                return calibrated_prob
            
            return raw_probability
        except Exception as e:
            logger.warning(f"Error calibrating probability: {str(e)}")
            return raw_probability
    
    def _extract_calibration_features(self, data):
        """Extract features for probability calibration"""
        try:
            if len(data) < 20:
                return None
                
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            # Technical indicators as features
            rsi = talib.RSI(close_prices, timeperiod=14)[-1]
            macd, macd_signal, _ = talib.MACD(close_prices)
            macd_diff = macd[-1] - macd_signal[-1]
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1]
            
            # Volume-based feature (if available)
            volume_ratio = 1.0
            if 'Volume' in data.columns:
                recent_volume = data['Volume'].tail(5).mean()
                avg_volume = data['Volume'].tail(20).mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            features = [rsi, macd_diff, atr, adx, volume_ratio]
            return features
        except Exception as e:
            logger.warning(f"Error extracting calibration features: {str(e)}")
            return None
    
    def _calculate_expected_value(self, probability, market_regime):
        """Calculate Expected Value: EV = (p * avg_win) - ((1-p) * avg_loss)"""
        try:
            # Get historical performance data
            avg_win = self.ev_tracker.get('avg_win', 20)  # Default 20 pips
            avg_loss = self.ev_tracker.get('avg_loss', 15)  # Default 15 pips
            
            # Adjust for market regime
            if market_regime.get('regime') == 'TRENDING':
                avg_win *= 1.2  # Higher wins in trending markets
            elif market_regime.get('regime') == 'RANGING':
                avg_win *= 0.8  # Lower wins in ranging markets
                avg_loss *= 1.1  # Higher losses in ranging markets
            
            # Calculate EV
            ev = (probability * avg_win) - ((1 - probability) * avg_loss)
            return ev
        except Exception as e:
            logger.warning(f"Error calculating expected value: {str(e)}")
            return 0
    
    def _get_dynamic_threshold(self, market_regime):
        """Get dynamic confidence threshold based on ADX regime"""
        try:
            adx = market_regime.get('adx', 0)
            
            if adx >= 40:
                return self.confidence_thresholds['strong_trend']['threshold']
            elif adx >= 25:
                return self.confidence_thresholds['moderate_trend']['threshold']
            else:
                return self.confidence_thresholds['weak_trend']['threshold']
        except Exception as e:
            logger.warning(f"Error getting dynamic threshold: {str(e)}")
            return 0.6  # Default threshold
    
    def _meta_labeling_decision(self, data, probability, market_regime):
        """Meta-labeling: decide whether to act on the signal"""
        try:
            if self.meta_classifier is None:
                # If no meta-classifier trained, use simple heuristics
                if probability > 0.6 and market_regime.get('adx', 0) > 20:
                    return 1  # Act on signal
                else:
                    return 0  # Don't act
            
            # Extract meta-features
            meta_features = self._extract_meta_features(data, probability, market_regime)
            if meta_features is not None:
                meta_features_scaled = self.scaler.transform([meta_features])
                decision = self.meta_classifier.predict(meta_features_scaled)[0]
                return decision
            
            return 0
        except Exception as e:
            logger.warning(f"Error in meta-labeling decision: {str(e)}")
            return 0
    
    def _extract_meta_features(self, data, probability, market_regime):
        """Extract features for meta-labeling"""
        try:
            # Meta-features: probability, market regime, volatility, time of day
            adx = market_regime.get('adx', 0)
            band_width = market_regime.get('band_width', 0)
            
            # Time-based feature (hour of day effect)
            current_hour = datetime.now().hour
            
            # Volatility feature
            if len(data) >= 20:
                returns = data['Close'].pct_change().tail(20)
                volatility = returns.std()
            else:
                volatility = 0
            
            meta_features = [probability, adx, band_width, current_hour, volatility]
            return meta_features
        except Exception as e:
            logger.warning(f"Error extracting meta-features: {str(e)}")
            return None
    
    def _calculate_kelly_fraction(self, probability, expected_value):
        """Calculate fractional Kelly sizing: f* = (b*p - (1-p))/b"""
        try:
            # Get win/loss ratio from tracker
            avg_win = self.ev_tracker.get('avg_win', 20)
            avg_loss = self.ev_tracker.get('avg_loss', 15)
            
            if avg_loss > 0:
                b = avg_win / avg_loss  # Win/loss ratio
                kelly_fraction = (b * probability - (1 - probability)) / b
                
                # Use fractional Kelly (25% of full Kelly) for safety
                fractional_kelly = 0.25 * max(0, kelly_fraction)
                
                # Cap at 2% for risk management
                return min(fractional_kelly, 0.02)
            
            return 0.01  # Default 1% position size
        except Exception as e:
            logger.warning(f"Error calculating Kelly fraction: {str(e)}")
            return 0.01
    
    def _determine_institutional_signal(self, weighted_signal, calibrated_probability, 
                                      market_regime, timeframe_consensus, price_action, 
                                      expected_value, dynamic_threshold, meta_decision):
        """Institutional-grade signal determination with all filters"""
        try:
            # Step 1: Check probability against dynamic threshold
            if calibrated_probability < dynamic_threshold:
                return 'HOLD', {'percentage': calibrated_probability * 100, 'status': 'BELOW_THRESHOLD'}
            
            # Step 2: Check Expected Value filter
            if expected_value <= 0:
                return 'HOLD', {'percentage': calibrated_probability * 100, 'status': 'NEGATIVE_EV'}
            
            # Step 3: Check meta-labeling decision
            if meta_decision == 0:
                return 'HOLD', {'percentage': calibrated_probability * 100, 'status': 'META_REJECT'}
            
            # Step 4: Determine signal direction
            if weighted_signal > 0.1:
                signal = 'BUY'
            elif weighted_signal < -0.1:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Step 5: Calculate final confidence
            base_confidence = calibrated_probability * 100
            
            # Boost confidence for strong market regimes
            if market_regime.get('adx', 0) > 30:
                base_confidence += 10
            
            # Boost for timeframe consensus
            if timeframe_consensus.get('consensus') == signal:
                base_confidence += 15
            
            # Reduce for price action risks
            if price_action.get('near_key_level'):
                base_confidence -= 10
            
            # Cap confidence
            final_confidence = min(100, max(0, base_confidence))
            
            # Determine status
            if final_confidence >= 85:
                status = 'INSTITUTIONAL_HIGH'
            elif final_confidence >= 70:
                status = 'INSTITUTIONAL_MEDIUM'
            elif final_confidence >= dynamic_threshold * 100:
                status = 'INSTITUTIONAL_LOW'
            else:
                status = 'INSTITUTIONAL_WEAK'
                signal = 'HOLD'
            
            return signal, {
                'percentage': round(final_confidence, 1),
                'status': status,
                'expected_value': expected_value,
                'dynamic_threshold': dynamic_threshold,
                'meta_decision': meta_decision,
                'filters_passed': {
                    'probability_threshold': calibrated_probability >= dynamic_threshold,
                    'positive_ev': expected_value > 0,
                    'meta_approval': meta_decision == 1
                }
            }
            
        except Exception as e:
            logger.warning(f"Error in institutional signal determination: {str(e)}")
            return 'HOLD', {'percentage': 0, 'status': 'ERROR'}

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