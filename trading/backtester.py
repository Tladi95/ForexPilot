import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstitutionalBacktester:
    """Institutional-Grade Backtesting with Walk-Forward Analysis and Triple-Barrier Labeling"""
    
    def __init__(self, ensemble_analyzer, risk_manager):
        self.ensemble_analyzer = ensemble_analyzer
        self.risk_manager = risk_manager
        
        # Triple-barrier parameters for intraday trading
        self.barrier_config = {
            '5m': {'profit_barrier': 0.0015, 'stop_barrier': 0.0008, 'time_barrier': 12},  # 1 hour
            '15m': {'profit_barrier': 0.0025, 'stop_barrier': 0.0012, 'time_barrier': 8},   # 2 hours
            '1h': {'profit_barrier': 0.004, 'stop_barrier': 0.002, 'time_barrier': 6}       # 6 hours
        }
        
        # Walk-forward parameters
        self.walk_forward_config = {
            'initial_window': 500,  # Initial training window
            'step_size': 50,        # Step size for walk-forward
            'min_trades': 20        # Minimum trades for valid period
        }
        
        # Performance tracking
        self.backtest_results = {}
        self.detailed_trades = []
        
    def run_walk_forward_backtest(self, data: pd.DataFrame, pair: str, timeframe: str = '15m', 
                                account_size: float = 10000) -> Dict:
        """
        Run comprehensive walk-forward backtest with institutional metrics
        
        Args:
            data: OHLCV data for backtesting
            pair: Currency pair (e.g., 'EURUSD')
            timeframe: Trading timeframe
            account_size: Account size for position sizing
            
        Returns:
            Comprehensive backtest results with institutional metrics
        """
        try:
            if len(data) < self.walk_forward_config['initial_window'] + 100:
                logger.warning(f"Insufficient data for walk-forward backtest: {len(data)} rows")
                return self._default_backtest_results()
            
            logger.info(f"Starting walk-forward backtest for {pair} on {timeframe}")
            
            # Prepare data
            data = data.sort_index().copy()
            
            # Initialize results tracking
            all_trades = []
            period_results = []
            
            # Walk-forward analysis
            initial_window = self.walk_forward_config['initial_window']
            step_size = self.walk_forward_config['step_size']
            
            for start_idx in range(initial_window, len(data) - 100, step_size):
                end_idx = min(start_idx + step_size, len(data))
                
                # Training data (for model calibration)
                train_data = data.iloc[start_idx - initial_window:start_idx]
                
                # Test data (for trading simulation)
                test_data = data.iloc[start_idx:end_idx]
                
                # Run period backtest
                period_trades = self._backtest_period(
                    train_data, test_data, pair, timeframe, account_size
                )
                
                if len(period_trades) >= self.walk_forward_config['min_trades']:
                    all_trades.extend(period_trades)
                    
                    # Calculate period metrics
                    period_metrics = self._calculate_period_metrics(period_trades)
                    period_metrics['period_start'] = test_data.index[0]
                    period_metrics['period_end'] = test_data.index[-1]
                    period_results.append(period_metrics)
            
            # Calculate comprehensive results
            if len(all_trades) > 0:
                results = self._calculate_comprehensive_results(
                    all_trades, period_results, pair, timeframe, account_size
                )
                
                # Save detailed results
                self._save_backtest_results(results, pair, timeframe)
                
                logger.info(f"Backtest completed: {len(all_trades)} trades, "
                          f"Win rate: {results['performance_metrics']['win_rate']:.1f}%")
                
                return results
            else:
                logger.warning("No valid trades generated in backtest")
                return self._default_backtest_results()
                
        except Exception as e:
            logger.error(f"Error in walk-forward backtest: {str(e)}")
            return self._default_backtest_results()
    
    def _backtest_period(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                        pair: str, timeframe: str, account_size: float) -> List[Dict]:
        """Backtest a specific period with triple-barrier labeling"""
        try:
            trades = []
            
            for i in range(len(test_data) - 50):  # Leave buffer for barriers
                current_data = pd.concat([train_data, test_data.iloc[:i+1]])
                
                if len(current_data) < 100:
                    continue
                
                # Get institutional signal
                analysis = self.ensemble_analyzer.analyze(current_data)
                
                if analysis['final_signal'] in ['BUY', 'SELL']:
                    # Create trade with triple-barrier labeling
                    trade = self._create_triple_barrier_trade(
                        test_data, i, analysis, pair, timeframe, account_size
                    )
                    
                    if trade:
                        trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.warning(f"Error in period backtest: {str(e)}")
            return []
    
    def _create_triple_barrier_trade(self, data: pd.DataFrame, entry_idx: int, 
                                   analysis: Dict, pair: str, timeframe: str, 
                                   account_size: float) -> Optional[Dict]:
        """Create trade with triple-barrier exit logic"""
        try:
            if entry_idx >= len(data) - 20:  # Need buffer for barriers
                return None
            
            entry_price = data['Close'].iloc[entry_idx]
            entry_time = data.index[entry_idx]
            signal = analysis['final_signal']
            
            # Get barrier parameters
            barriers = self.barrier_config.get(timeframe, self.barrier_config['15m'])
            
            # Calculate barrier levels
            if signal == 'BUY':
                profit_target = entry_price * (1 + barriers['profit_barrier'])
                stop_loss = entry_price * (1 - barriers['stop_barrier'])
            else:  # SELL
                profit_target = entry_price * (1 - barriers['profit_barrier'])
                stop_loss = entry_price * (1 + barriers['stop_barrier'])
            
            # Find exit point
            exit_info = self._find_triple_barrier_exit(
                data, entry_idx, profit_target, stop_loss, 
                barriers['time_barrier'], signal
            )
            
            if not exit_info:
                return None
            
            # Calculate trade results
            if signal == 'BUY':
                pnl_pips = (exit_info['exit_price'] - entry_price) / 0.0001  # Assuming major pair
            else:
                pnl_pips = (entry_price - exit_info['exit_price']) / 0.0001
            
            # Get position sizing from risk manager
            risk_calc = self.risk_manager.calculate_risk_levels(
                pair=pair,
                current_price=entry_price,
                account_size=account_size,
                risk_percent=0.5,  # Conservative for backtesting
                timeframe=timeframe,
                kelly_fraction=analysis.get('kelly_fraction')
            )
            
            position_size = risk_calc['position_size']['micro_lots']
            pnl_usd = pnl_pips * position_size * 0.1  # Approximate USD per pip for micro lot
            
            return {
                'entry_time': entry_time,
                'exit_time': exit_info['exit_time'],
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_info['exit_price'],
                'exit_reason': exit_info['exit_reason'],
                'pnl_pips': round(pnl_pips, 1),
                'pnl_usd': round(pnl_usd, 2),
                'position_size': position_size,
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'confidence': analysis['confidence']['percentage'],
                'expected_value': analysis.get('expected_value', 0),
                'kelly_fraction': analysis.get('kelly_fraction', 0),
                'market_regime': analysis['market_regime']['regime'],
                'meta_decision': analysis.get('meta_decision', 1),
                'filters_passed': analysis.get('filters_passed', {}),
                'duration_bars': exit_info['duration_bars']
            }
            
        except Exception as e:
            logger.warning(f"Error creating triple-barrier trade: {str(e)}")
            return None
    
    def _find_triple_barrier_exit(self, data: pd.DataFrame, entry_idx: int, 
                                 profit_target: float, stop_loss: float, 
                                 time_barrier: int, signal: str) -> Optional[Dict]:
        """Find exit point using triple-barrier logic"""
        try:
            max_idx = min(entry_idx + time_barrier, len(data) - 1)
            
            for i in range(entry_idx + 1, max_idx + 1):
                current_high = data['High'].iloc[i]
                current_low = data['Low'].iloc[i]
                current_close = data['Close'].iloc[i]
                
                if signal == 'BUY':
                    # Check profit target
                    if current_high >= profit_target:
                        return {
                            'exit_price': profit_target,
                            'exit_time': data.index[i],
                            'exit_reason': 'profit_target',
                            'duration_bars': i - entry_idx
                        }
                    # Check stop loss
                    if current_low <= stop_loss:
                        return {
                            'exit_price': stop_loss,
                            'exit_time': data.index[i],
                            'exit_reason': 'stop_loss',
                            'duration_bars': i - entry_idx
                        }
                else:  # SELL
                    # Check profit target
                    if current_low <= profit_target:
                        return {
                            'exit_price': profit_target,
                            'exit_time': data.index[i],
                            'exit_reason': 'profit_target',
                            'duration_bars': i - entry_idx
                        }
                    # Check stop loss
                    if current_high >= stop_loss:
                        return {
                            'exit_price': stop_loss,
                            'exit_time': data.index[i],
                            'exit_reason': 'stop_loss',
                            'duration_bars': i - entry_idx
                        }
            
            # Time barrier hit
            return {
                'exit_price': data['Close'].iloc[max_idx],
                'exit_time': data.index[max_idx],
                'exit_reason': 'time_barrier',
                'duration_bars': max_idx - entry_idx
            }
            
        except Exception as e:
            logger.warning(f"Error finding triple-barrier exit: {str(e)}")
            return None
    
    def _calculate_period_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics for a specific period"""
        try:
            if not trades:
                return {'trades': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0}
            
            winning_trades = [t for t in trades if t['pnl_usd'] > 0]
            losing_trades = [t for t in trades if t['pnl_usd'] < 0]
            
            total_profit = sum(t['pnl_usd'] for t in winning_trades)
            total_loss = abs(sum(t['pnl_usd'] for t in losing_trades))
            
            win_rate = len(winning_trades) / len(trades) * 100
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate Sharpe ratio
            returns = [t['pnl_usd'] for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            return {
                'trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': round(win_rate, 1),
                'profit_factor': round(profit_factor, 2),
                'total_profit': round(total_profit, 2),
                'total_loss': round(total_loss, 2),
                'net_profit': round(total_profit - total_loss, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'avg_win': round(total_profit / len(winning_trades), 2) if winning_trades else 0,
                'avg_loss': round(total_loss / len(losing_trades), 2) if losing_trades else 0
            }
            
        except Exception as e:
            logger.warning(f"Error calculating period metrics: {str(e)}")
            return {'trades': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0}
    
    def _calculate_comprehensive_results(self, all_trades: List[Dict], 
                                       period_results: List[Dict], pair: str, 
                                       timeframe: str, account_size: float) -> Dict:
        """Calculate comprehensive institutional-grade results"""
        try:
            # Overall performance metrics
            overall_metrics = self._calculate_period_metrics(all_trades)
            
            # Confidence-based analysis
            confidence_analysis = self._analyze_by_confidence(all_trades)
            
            # Regime-based analysis
            regime_analysis = self._analyze_by_regime(all_trades)
            
            # Expected Value analysis
            ev_analysis = self._analyze_expected_value(all_trades)
            
            # Meta-labeling effectiveness
            meta_analysis = self._analyze_meta_labeling(all_trades)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(all_trades, account_size)
            
            # Trade frequency and expectancy
            frequency_metrics = self._calculate_frequency_metrics(all_trades, timeframe)
            
            return {
                'backtest_info': {
                    'pair': pair,
                    'timeframe': timeframe,
                    'account_size': account_size,
                    'total_periods': len(period_results),
                    'backtest_date': datetime.now().isoformat(),
                    'walk_forward_config': self.walk_forward_config,
                    'barrier_config': self.barrier_config[timeframe]
                },
                'performance_metrics': overall_metrics,
                'confidence_distribution': confidence_analysis,
                'regime_performance': regime_analysis,
                'expected_value_analysis': ev_analysis,
                'meta_labeling_analysis': meta_analysis,
                'risk_metrics': risk_metrics,
                'frequency_metrics': frequency_metrics,
                'period_results': period_results,
                'detailed_trades': all_trades[-100:],  # Last 100 trades for analysis
                'summary': {
                    'recommendation': self._generate_recommendation(overall_metrics, risk_metrics),
                    'key_insights': self._generate_insights(all_trades, overall_metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive results: {str(e)}")
            return self._default_backtest_results()
    
    def _analyze_by_confidence(self, trades: List[Dict]) -> Dict:
        """Analyze performance by confidence buckets"""
        try:
            buckets = {'60-70%': [], '70-80%': [], '80-90%': [], '90-100%': []}
            
            for trade in trades:
                confidence = trade.get('confidence', 50)
                if 60 <= confidence < 70:
                    buckets['60-70%'].append(trade)
                elif 70 <= confidence < 80:
                    buckets['70-80%'].append(trade)
                elif 80 <= confidence < 90:
                    buckets['80-90%'].append(trade)
                elif confidence >= 90:
                    buckets['90-100%'].append(trade)
            
            analysis = {}
            for bucket, bucket_trades in buckets.items():
                if bucket_trades:
                    metrics = self._calculate_period_metrics(bucket_trades)
                    analysis[bucket] = {
                        'trades': len(bucket_trades),
                        'win_rate': metrics['win_rate'],
                        'avg_pnl': round(np.mean([t['pnl_usd'] for t in bucket_trades]), 2),
                        'sharpe_ratio': metrics['sharpe_ratio']
                    }
                else:
                    analysis[bucket] = {'trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'sharpe_ratio': 0}
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing by confidence: {str(e)}")
            return {}
    
    def _analyze_by_regime(self, trades: List[Dict]) -> Dict:
        """Analyze performance by market regime"""
        try:
            regimes = {}
            
            for trade in trades:
                regime = trade.get('market_regime', 'UNKNOWN')
                if regime not in regimes:
                    regimes[regime] = []
                regimes[regime].append(trade)
            
            analysis = {}
            for regime, regime_trades in regimes.items():
                if regime_trades:
                    metrics = self._calculate_period_metrics(regime_trades)
                    analysis[regime] = {
                        'trades': len(regime_trades),
                        'win_rate': metrics['win_rate'],
                        'profit_factor': metrics['profit_factor'],
                        'avg_pnl': round(np.mean([t['pnl_usd'] for t in regime_trades]), 2)
                    }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing by regime: {str(e)}")
            return {}
    
    def _analyze_expected_value(self, trades: List[Dict]) -> Dict:
        """Analyze Expected Value effectiveness"""
        try:
            ev_positive = [t for t in trades if t.get('expected_value', 0) > 0]
            ev_negative = [t for t in trades if t.get('expected_value', 0) <= 0]
            
            return {
                'positive_ev': {
                    'trades': len(ev_positive),
                    'win_rate': self._calculate_period_metrics(ev_positive)['win_rate'] if ev_positive else 0,
                    'avg_pnl': round(np.mean([t['pnl_usd'] for t in ev_positive]), 2) if ev_positive else 0
                },
                'negative_ev': {
                    'trades': len(ev_negative),
                    'win_rate': self._calculate_period_metrics(ev_negative)['win_rate'] if ev_negative else 0,
                    'avg_pnl': round(np.mean([t['pnl_usd'] for t in ev_negative]), 2) if ev_negative else 0
                }
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing expected value: {str(e)}")
            return {}
    
    def _analyze_meta_labeling(self, trades: List[Dict]) -> Dict:
        """Analyze meta-labeling effectiveness"""
        try:
            meta_approved = [t for t in trades if t.get('meta_decision', 1) == 1]
            meta_rejected = [t for t in trades if t.get('meta_decision', 1) == 0]
            
            return {
                'meta_approved': {
                    'trades': len(meta_approved),
                    'win_rate': self._calculate_period_metrics(meta_approved)['win_rate'] if meta_approved else 0,
                    'avg_pnl': round(np.mean([t['pnl_usd'] for t in meta_approved]), 2) if meta_approved else 0
                },
                'meta_rejected': {
                    'trades': len(meta_rejected),
                    'win_rate': self._calculate_period_metrics(meta_rejected)['win_rate'] if meta_rejected else 0,
                    'avg_pnl': round(np.mean([t['pnl_usd'] for t in meta_rejected]), 2) if meta_rejected else 0
                },
                'effectiveness': len(meta_approved) / (len(meta_approved) + len(meta_rejected)) * 100 if trades else 0
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing meta-labeling: {str(e)}")
            return {}
    
    def _calculate_risk_metrics(self, trades: List[Dict], account_size: float) -> Dict:
        """Calculate comprehensive risk metrics"""
        try:
            if not trades:
                return {}
            
            returns = [t['pnl_usd'] / account_size for t in trades]
            cumulative_returns = np.cumprod(1 + np.array(returns))
            
            # Maximum Drawdown
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = abs(np.min(drawdown))
            
            # Calmar Ratio
            total_return = cumulative_returns[-1] - 1
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino Ratio
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0
            sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean([r for r in returns if r <= var_95])
            
            return {
                'max_drawdown': round(max_drawdown, 4),
                'calmar_ratio': round(calmar_ratio, 2),
                'sortino_ratio': round(sortino_ratio, 2),
                'var_95': round(var_95, 4),
                'cvar_95': round(cvar_95, 4),
                'volatility': round(np.std(returns), 4),
                'skewness': round(float(pd.Series(returns).skew()), 2),
                'kurtosis': round(float(pd.Series(returns).kurtosis()), 2)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _calculate_frequency_metrics(self, trades: List[Dict], timeframe: str) -> Dict:
        """Calculate trade frequency and expectancy metrics"""
        try:
            if not trades:
                return {}
            
            # Calculate time span
            start_time = min(trade['entry_time'] for trade in trades)
            end_time = max(trade['exit_time'] for trade in trades)
            total_hours = (end_time - start_time).total_seconds() / 3600
            
            # Trade frequency
            trades_per_day = len(trades) / (total_hours / 24) if total_hours > 0 else 0
            
            # Average trade duration
            avg_duration = np.mean([t['duration_bars'] for t in trades])
            
            # Expectancy
            win_rate = len([t for t in trades if t['pnl_usd'] > 0]) / len(trades)
            avg_win = np.mean([t['pnl_usd'] for t in trades if t['pnl_usd'] > 0])
            avg_loss = abs(np.mean([t['pnl_usd'] for t in trades if t['pnl_usd'] < 0]))
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            return {
                'trades_per_day': round(trades_per_day, 1),
                'avg_duration_bars': round(avg_duration, 1),
                'expectancy_per_trade': round(expectancy, 2),
                'expectancy_per_day': round(expectancy * trades_per_day, 2),
                'total_trading_days': round(total_hours / 24, 1)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating frequency metrics: {str(e)}")
            return {}
    
    def _generate_recommendation(self, performance: Dict, risk: Dict) -> str:
        """Generate trading recommendation based on backtest results"""
        try:
            win_rate = performance.get('win_rate', 0)
            profit_factor = performance.get('profit_factor', 0)
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            max_drawdown = risk.get('max_drawdown', 1)
            
            if win_rate >= 60 and profit_factor >= 1.5 and sharpe_ratio >= 1.0 and max_drawdown <= 0.15:
                return "STRONG BUY - Excellent performance across all metrics"
            elif win_rate >= 55 and profit_factor >= 1.2 and sharpe_ratio >= 0.5 and max_drawdown <= 0.25:
                return "BUY - Good performance with acceptable risk"
            elif win_rate >= 50 and profit_factor >= 1.0 and max_drawdown <= 0.35:
                return "NEUTRAL - Mixed performance, requires optimization"
            else:
                return "AVOID - Poor performance or excessive risk"
                
        except Exception:
            return "INSUFFICIENT DATA - Unable to generate recommendation"
    
    def _generate_insights(self, trades: List[Dict], performance: Dict) -> List[str]:
        """Generate key insights from backtest results"""
        insights = []
        
        try:
            # Performance insights
            win_rate = performance.get('win_rate', 0)
            if win_rate > 65:
                insights.append(f"Excellent win rate of {win_rate}% indicates strong signal quality")
            elif win_rate < 45:
                insights.append(f"Low win rate of {win_rate}% suggests signal optimization needed")
            
            # Exit reason analysis
            exit_reasons = {}
            for trade in trades:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            total_exits = sum(exit_reasons.values())
            if total_exits > 0:
                for reason, count in exit_reasons.items():
                    pct = (count / total_exits) * 100
                    if pct > 30:
                        insights.append(f"{pct:.0f}% of trades exit via {reason.replace('_', ' ')}")
            
            # Regime performance
            regime_trades = {}
            for trade in trades:
                regime = trade.get('market_regime', 'UNKNOWN')
                if regime not in regime_trades:
                    regime_trades[regime] = []
                regime_trades[regime].append(trade['pnl_usd'])
            
            for regime, pnls in regime_trades.items():
                if len(pnls) > 10:  # Sufficient sample size
                    avg_pnl = np.mean(pnls)
                    if avg_pnl > 5:
                        insights.append(f"Strong performance in {regime} markets (avg: ${avg_pnl:.1f})")
                    elif avg_pnl < -2:
                        insights.append(f"Weak performance in {regime} markets (avg: ${avg_pnl:.1f})")
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.warning(f"Error generating insights: {str(e)}")
            return ["Analysis completed successfully"]
    
    def _save_backtest_results(self, results: Dict, pair: str, timeframe: str):
        """Save backtest results to file"""
        try:
            filename = f"backtest_{pair}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join('backtest_results', filename)
            
            # Create directory if it doesn't exist
            os.makedirs('backtest_results', exist_ok=True)
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Error saving backtest results: {str(e)}")
    
    def _default_backtest_results(self) -> Dict:
        """Return default backtest results when analysis fails"""
        return {
            'backtest_info': {
                'pair': 'UNKNOWN',
                'timeframe': 'UNKNOWN',
                'status': 'FAILED',
                'backtest_date': datetime.now().isoformat()
            },
            'performance_metrics': {
                'trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'net_profit': 0
            },
            'confidence_distribution': {},
            'regime_performance': {},
            'risk_metrics': {},
            'summary': {
                'recommendation': 'INSUFFICIENT DATA',
                'key_insights': ['Backtest failed - insufficient data or configuration error']
            }
        }
    
    def get_backtest_summary(self, pair: str = None, timeframe: str = None) -> Dict:
        """Get summary of all backtest results"""
        try:
            backtest_dir = 'backtest_results'
            if not os.path.exists(backtest_dir):
                return {'status': 'No backtest results found'}
            
            summaries = []
            for filename in os.listdir(backtest_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(backtest_dir, filename), 'r') as f:
                            result = json.load(f)
                            
                        # Filter by pair and timeframe if specified
                        if pair and result['backtest_info'].get('pair') != pair:
                            continue
                        if timeframe and result['backtest_info'].get('timeframe') != timeframe:
                            continue
                        
                        summary = {
                            'filename': filename,
                            'pair': result['backtest_info'].get('pair'),
                            'timeframe': result['backtest_info'].get('timeframe'),
                            'date': result['backtest_info'].get('backtest_date'),
                            'trades': result['performance_metrics'].get('trades', 0),
                            'win_rate': result['performance_metrics'].get('win_rate', 0),
                            'net_profit': result['performance_metrics'].get('net_profit', 0),
                            'max_drawdown': result.get('risk_metrics', {}).get('max_drawdown', 0),
                            'recommendation': result.get('summary', {}).get('recommendation', 'UNKNOWN')
                        }
                        summaries.append(summary)
                        
                    except Exception as e:
                        logger.warning(f"Error reading backtest file {filename}: {str(e)}")
                        continue
            
            return {
                'status': 'success',
                'total_backtests': len(summaries),
                'summaries': sorted(summaries, key=lambda x: x['date'], reverse=True)
            }
            
        except Exception as e:
            logger.error(f"Error getting backtest summary: {str(e)}")
            return {'status': 'error', 'message': str(e)}