import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskManager:
    """Institutional-Grade Risk Management with VaR/CVaR, Kelly Sizing, and Intraday Optimization"""
    
    def __init__(self):
        # Pip values for different currency pairs (in USD for 1 standard lot)
        self.pip_values = {
            'EURUSD': 10.0,  # $10 per pip for 1 standard lot
            'USDJPY': 10.0,  # Approximately $10 per pip (varies with JPY rate)
            'GBPUSD': 10.0,
            'AUDUSD': 10.0,
            'USDCAD': 10.0,
            'USDCHF': 10.0
        }
        
        # Intraday stop loss pips for 5m-15m timeframes (scaled down)
        self.intraday_stop_loss_pips = {
            'EURUSD': {'5m': 5, '15m': 8, '1h': 15},
            'USDJPY': {'5m': 7, '15m': 12, '1h': 20},
            'GBPUSD': {'5m': 6, '15m': 10, '1h': 18},
            'AUDUSD': {'5m': 5, '15m': 9, '1h': 16},
            'USDCAD': {'5m': 5, '15m': 9, '1h': 16},
            'USDCHF': {'5m': 5, '15m': 8, '1h': 15}
        }
        
        # Intraday take profit pips (2:1 ratios maintained)
        self.intraday_take_profit_pips = {
            'EURUSD': {'5m': 10, '15m': 16, '1h': 30},
            'USDJPY': {'5m': 14, '15m': 24, '1h': 40},
            'GBPUSD': {'5m': 12, '15m': 20, '1h': 36},
            'AUDUSD': {'5m': 10, '15m': 18, '1h': 32},
            'USDCAD': {'5m': 10, '15m': 18, '1h': 32},
            'USDCHF': {'5m': 10, '15m': 16, '1h': 30}
        }
        
        # Risk budgets for intraday trading
        self.intraday_risk_limits = {
            'per_trade': 0.5,  # 0.5% per trade (reduced for higher frequency)
            'per_session': 2.0,  # 2% per trading session
            'daily_max': 3.0     # 3% daily maximum
        }
        
        # VaR/CVaR parameters
        self.var_confidence = 0.95  # 95% confidence level
        self.var_lookback = 100     # 100-period lookback
        
        # Performance tracking for risk metrics
        self.performance_tracker = self._load_performance_data()
        
        # Session risk tracking
        self.session_risk = {
            'current_exposure': 0.0,
            'trades_today': 0,
            'session_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': 0.0
        }
    
    def calculate_risk_levels(self, pair, current_price, account_size=10000.0, risk_percent=1.5, atr_data=None, 
                            timeframe='15m', kelly_fraction=None, volatility_cap=None):
        """
        Institutional-grade risk calculation with Kelly sizing, VaR limits, and intraday optimization
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            current_price (float): Current market price
            account_size (float): Account balance in USD
            risk_percent (float): Base risk percentage per trade
            atr_data (dict): ATR data from ensemble analysis (optional)
            timeframe (str): Trading timeframe ('5m', '15m', '1h')
            kelly_fraction (float): Kelly fraction from ensemble analyzer
            volatility_cap (float): Volatility-based position size cap
        
        Returns:
            dict: Comprehensive institutional risk management data
        """
        try:
            # Calculate pip size based on currency pair
            if 'JPY' in pair:
                pip_size = 0.01  # For JPY pairs, 1 pip = 0.01
            else:
                pip_size = 0.0001  # For other pairs, 1 pip = 0.0001
            
            # Get intraday-optimized pip values
            timeframe = timeframe if timeframe in ['5m', '15m', '1h'] else '15m'
            
            # Use ATR-based calculations if available, otherwise use intraday pips
            if atr_data and atr_data.get('value', 0) > 0:
                # ATR-based dynamic levels (scaled for intraday)
                atr_value = atr_data['value']
                intraday_multiplier = {'5m': 1.0, '15m': 1.2, '1h': 1.5}[timeframe]
                
                sl_distance = atr_value * 1.5 * intraday_multiplier
                tp_distance = atr_value * 3.0 * intraday_multiplier
                
                sl_pips = sl_distance / pip_size
                tp_pips = tp_distance / pip_size
                
                calculation_method = f'ATR-based ({timeframe})'
            else:
                # Use intraday-optimized fixed pips
                sl_pips = self.intraday_stop_loss_pips.get(pair, {}).get(timeframe, 8)
                tp_pips = self.intraday_take_profit_pips.get(pair, {}).get(timeframe, 16)
                sl_distance = sl_pips * pip_size
                tp_distance = tp_pips * pip_size
                
                calculation_method = f'Intraday-optimized ({timeframe})'
            
            # Get pip value for position sizing
            pip_value = self.pip_values.get(pair, 10.0)
            
            # Institutional position sizing with Kelly fraction and risk limits
            base_risk_percent = min(risk_percent, self.intraday_risk_limits['per_trade'])
            
            # Apply Kelly fraction if provided
            if kelly_fraction and kelly_fraction > 0:
                adjusted_risk_percent = min(base_risk_percent, kelly_fraction * 100)
            else:
                adjusted_risk_percent = base_risk_percent
            
            # Check session risk limits
            if self._check_session_risk_limits(account_size, adjusted_risk_percent):
                adjusted_risk_percent *= 0.5  # Reduce position size if approaching limits
            
            risk_amount = account_size * (adjusted_risk_percent / 100)
            
            # Calculate base position size
            position_size_lots = risk_amount / (sl_pips * pip_value)
            
            # Apply volatility cap if provided
            if volatility_cap and volatility_cap > 0:
                volatility_adjusted_size = min(position_size_lots, volatility_cap)
                position_size_lots = volatility_adjusted_size
            
            # Convert to micro lots for practical trading
            micro_lots = position_size_lots * 100  # 1 standard lot = 100 micro lots
            micro_lots = max(0.01, round(micro_lots, 2))  # Minimum 0.01 micro lots
            
            # Calculate actual risk with position size
            actual_risk = (micro_lots / 100) * sl_pips * pip_value
            
            # Calculate VaR and CVaR
            var_metrics = self._calculate_var_cvar(actual_risk, pair)
            
            # Calculate Calmar ratio and max drawdown
            performance_metrics = self._calculate_performance_metrics()
            
            # Calculate stop loss and take profit prices using actual distances
            stop_loss_buy = round(current_price - sl_distance, 5)
            stop_loss_sell = round(current_price + sl_distance, 5)
            take_profit_buy = round(current_price + tp_distance, 5)
            take_profit_sell = round(current_price - tp_distance, 5)
            
            return {
                'stop_loss': {
                    'buy': stop_loss_buy,
                    'sell': stop_loss_sell,
                    'pips': round(sl_pips, 1),
                    'distance': round(sl_distance, 5)
                },
                'take_profit': {
                    'buy': take_profit_buy,
                    'sell': take_profit_sell,
                    'pips': round(tp_pips, 1),
                    'distance': round(tp_distance, 5)
                },
                'position_size': {
                    'micro_lots': micro_lots,
                    'standard_lots': round(position_size_lots, 4),
                    'units': int(micro_lots * 1000),
                    'kelly_adjusted': kelly_fraction is not None,
                    'volatility_capped': volatility_cap is not None
                },
                'risk_metrics': {
                    'risk_amount': round(actual_risk, 2),
                    'risk_percent': round((actual_risk / account_size) * 100, 2),
                    'base_risk_percent': base_risk_percent,
                    'adjusted_risk_percent': adjusted_risk_percent,
                    'var_95': var_metrics['var_95'],
                    'cvar_95': var_metrics['cvar_95'],
                    'max_drawdown': performance_metrics['max_drawdown'],
                    'calmar_ratio': performance_metrics['calmar_ratio']
                },
                'session_limits': {
                    'current_exposure': self.session_risk['current_exposure'],
                    'trades_today': self.session_risk['trades_today'],
                    'session_pnl': self.session_risk['session_pnl'],
                    'daily_limit_used': (self.session_risk['current_exposure'] / account_size) * 100
                },
                'reward_risk_ratio': round(tp_pips / sl_pips, 2) if sl_pips > 0 else 2.0,
                'pip_value': pip_value,
                'pip_size': pip_size,
                'timeframe': timeframe,
                'calculation_method': calculation_method,
                'institutional_features': {
                    'atr_used': atr_data is not None and atr_data.get('value', 0) > 0,
                    'kelly_sizing': kelly_fraction is not None,
                    'volatility_cap': volatility_cap is not None,
                    'session_limits': True,
                    'var_monitoring': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk levels for {pair}: {str(e)}")
            return self._default_risk_calculation(current_price)
    
    def _load_performance_data(self):
        """Load historical performance data for risk calculations"""
        try:
            if os.path.exists('risk_performance.json'):
                with open('risk_performance.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading performance data: {str(e)}")
        
        # Initialize performance data
        return {
            'daily_returns': [],
            'trade_pnl': [],
            'drawdowns': [],
            'peak_equity': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0
        }
    
    def _check_session_risk_limits(self, account_size, risk_percent):
        """Check if trade would breach session risk limits"""
        try:
            # Calculate potential new exposure
            potential_risk = account_size * (risk_percent / 100)
            new_exposure = self.session_risk['current_exposure'] + potential_risk
            
            # Check various limits
            session_limit = account_size * (self.intraday_risk_limits['per_session'] / 100)
            daily_limit = account_size * (self.intraday_risk_limits['daily_max'] / 100)
            
            # Return True if approaching limits (>80% of limit)
            return (new_exposure > session_limit * 0.8) or (new_exposure > daily_limit * 0.8)
        except Exception as e:
            logger.warning(f"Error checking session risk limits: {str(e)}")
            return False
    
    def _calculate_var_cvar(self, position_risk, pair):
        """Calculate Value at Risk and Conditional Value at Risk"""
        try:
            # Get historical returns for the pair
            returns = self.performance_tracker.get('daily_returns', [])
            
            if len(returns) < 30:  # Need minimum data
                return {
                    'var_95': position_risk * 2.0,  # Conservative estimate
                    'cvar_95': position_risk * 3.0,
                    'var_99': position_risk * 2.5,
                    'data_points': len(returns)
                }
            
            # Convert to numpy array and calculate percentiles
            returns_array = np.array(returns[-self.var_lookback:])
            
            # Calculate VaR at different confidence levels
            var_95 = np.percentile(returns_array, 5)  # 5th percentile for 95% VaR
            var_99 = np.percentile(returns_array, 1)  # 1st percentile for 99% VaR
            
            # Calculate CVaR (Expected Shortfall)
            # Average of returns below VaR threshold
            tail_returns_95 = returns_array[returns_array <= var_95]
            cvar_95 = np.mean(tail_returns_95) if len(tail_returns_95) > 0 else var_95
            
            tail_returns_99 = returns_array[returns_array <= var_99]
            cvar_99 = np.mean(tail_returns_99) if len(tail_returns_99) > 0 else var_99
            
            # Scale by position size
            return {
                'var_95': abs(var_95 * position_risk),
                'var_99': abs(var_99 * position_risk),
                'cvar_95': abs(cvar_95 * position_risk),
                'cvar_99': abs(cvar_99 * position_risk),
                'data_points': len(returns_array)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating VaR/CVaR: {str(e)}")
            return {
                'var_95': position_risk * 2.0,
                'cvar_95': position_risk * 3.0,
                'var_99': position_risk * 2.5,
                'cvar_99': position_risk * 3.5,
                'data_points': 0
            }
    
    def _calculate_performance_metrics(self):
        """Calculate rolling performance metrics including Calmar ratio"""
        try:
            returns = self.performance_tracker.get('daily_returns', [])
            
            if len(returns) < 10:
                return {
                    'max_drawdown': 0.0,
                    'calmar_ratio': 0.0,
                    'sharpe_ratio': 0.0,
                    'volatility': 0.0,
                    'total_return': 0.0
                }
            
            returns_array = np.array(returns)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + returns_array)
            
            # Calculate running maximum (peak equity)
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # Update session tracking
            self.session_risk['max_drawdown'] = max_drawdown
            self.session_risk['peak_equity'] = running_max[-1] if len(running_max) > 0 else 0.0
            
            # Calculate annualized metrics
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0.0
            
            # Annualized Sharpe ratio (assuming 252 trading days)
            sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0.0
            
            # Calmar ratio (annualized return / max drawdown)
            calmar_ratio = (total_return * 252 / len(returns)) / max_drawdown if max_drawdown > 0 else 0.0
            
            return {
                'max_drawdown': round(max_drawdown, 4),
                'calmar_ratio': round(calmar_ratio, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'volatility': round(std_return * np.sqrt(252), 4),
                'total_return': round(total_return, 4),
                'current_drawdown': round(drawdowns[-1], 4) if len(drawdowns) > 0 else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {str(e)}")
            return {
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'sharpe_ratio': 0.0,
                'volatility': 0.0,
                'total_return': 0.0,
                'current_drawdown': 0.0
            }
    
    def update_session_risk(self, trade_pnl, account_size):
        """Update session risk tracking with new trade"""
        try:
            self.session_risk['trades_today'] += 1
            self.session_risk['session_pnl'] += trade_pnl
            
            # Update performance tracking
            daily_return = trade_pnl / account_size
            self.performance_tracker['daily_returns'].append(daily_return)
            self.performance_tracker['trade_pnl'].append(trade_pnl)
            
            # Keep only last 252 days (1 year)
            if len(self.performance_tracker['daily_returns']) > 252:
                self.performance_tracker['daily_returns'].pop(0)
                self.performance_tracker['trade_pnl'].pop(0)
            
            # Save performance data
            try:
                with open('risk_performance.json', 'w') as f:
                    json.dump(self.performance_tracker, f, indent=2)
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Error updating session risk: {str(e)}")
    
    def get_risk_summary(self, account_size):
        """Get comprehensive risk summary for dashboard"""
        try:
            performance_metrics = self._calculate_performance_metrics()
            
            return {
                'session_metrics': {
                    'trades_today': self.session_risk['trades_today'],
                    'session_pnl': round(self.session_risk['session_pnl'], 2),
                    'current_exposure': round(self.session_risk['current_exposure'], 2),
                    'exposure_percent': round((self.session_risk['current_exposure'] / account_size) * 100, 2)
                },
                'risk_limits': {
                    'per_trade_limit': self.intraday_risk_limits['per_trade'],
                    'session_limit': self.intraday_risk_limits['per_session'],
                    'daily_limit': self.intraday_risk_limits['daily_max'],
                    'session_limit_used': round((self.session_risk['current_exposure'] / (account_size * self.intraday_risk_limits['per_session'] / 100)) * 100, 1)
                },
                'performance_metrics': performance_metrics,
                'var_summary': {
                    'confidence_level': self.var_confidence,
                    'lookback_periods': self.var_lookback,
                    'data_points': len(self.performance_tracker.get('daily_returns', []))
                }
            }
        except Exception as e:
            logger.warning(f"Error getting risk summary: {str(e)}")
            return {}
    
    def _default_risk_calculation(self, current_price):
        """Return default risk calculation when error occurs"""
        return {
            'stop_loss': {'buy': current_price * 0.998, 'sell': current_price * 1.002, 'pips': 8},
            'take_profit': {'buy': current_price * 1.004, 'sell': current_price * 0.996, 'pips': 16},
            'position_size': {'micro_lots': 0.01, 'standard_lots': 0.0001, 'units': 10},
            'risk_metrics': {
                'risk_amount': 0,
                'risk_percent': 0,
                'var_95': 0,
                'cvar_95': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0
            },
            'session_limits': {
                'current_exposure': 0,
                'trades_today': 0,
                'session_pnl': 0,
                'daily_limit_used': 0
            },
            'reward_risk_ratio': 2.0,
            'pip_value': 10.0,
            'pip_size': 0.0001,
            'timeframe': '15m',
            'calculation_method': 'Default (Error)',
            'institutional_features': {
                'atr_used': False,
                'kelly_sizing': False,
                'volatility_cap': False,
                'session_limits': True,
                'var_monitoring': True
            }
        }