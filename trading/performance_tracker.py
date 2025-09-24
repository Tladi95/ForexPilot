import json
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Performance tracking system for trading strategy"""
    
    def __init__(self, data_file='trading_performance.json'):
        self.data_file = data_file
        self.performance_data = self._load_performance_data()
    
    def _load_performance_data(self):
        """Load performance data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded performance data: {len(data.get('trades', []))} trades")
                    return data
            else:
                return self._initialize_performance_data()
        except Exception as e:
            logger.warning(f"Error loading performance data: {str(e)}")
            return self._initialize_performance_data()
    
    def _initialize_performance_data(self):
        """Initialize empty performance data structure"""
        return {
            'trades': [],
            'statistics': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'current_streak': 0,
                'best_streak': 0,
                'worst_streak': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            },
            'last_updated': datetime.now().isoformat(),
            'system_start_date': datetime.now().isoformat()
        }
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            self.performance_data['last_updated'] = datetime.now().isoformat()
            with open(self.data_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
            logger.info("Performance data saved successfully")
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
    
    def record_signal(self, pair, signal, confidence, entry_price, sl_price, tp_price):
        """
        Record a trading signal (for simulation/tracking purposes)
        
        Args:
            pair (str): Currency pair
            signal (str): BUY/SELL/HOLD
            confidence (dict): Confidence data
            entry_price (float): Entry price
            sl_price (float): Stop loss price
            tp_price (float): Take profit price
        """
        try:
            if signal in ['BUY', 'SELL']:
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'pair': pair,
                    'signal': signal,
                    'confidence': confidence,
                    'entry_price': entry_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'status': 'OPEN',
                    'exit_price': None,
                    'profit_loss': None,
                    'result': None
                }
                
                self.performance_data['trades'].append(trade)
                logger.info(f"Recorded {signal} signal for {pair} at {entry_price}")
                
                # For demonstration purposes, we'll simulate some trade outcomes
                # In a real system, this would be updated when trades actually close
                self._simulate_trade_outcome(len(self.performance_data['trades']) - 1)
                
        except Exception as e:
            logger.error(f"Error recording signal: {str(e)}")
    
    def _simulate_trade_outcome(self, trade_index):
        """
        Simulate trade outcomes for demonstration (replace with real trade tracking)
        """
        try:
            trade = self.performance_data['trades'][trade_index]
            
            # Simple simulation: 65% win rate based on confidence
            import random
            confidence_pct = trade['confidence'].get('percentage', 50)
            
            # Higher confidence = higher win probability
            win_probability = min(0.85, confidence_pct / 100 + 0.15)
            
            if random.random() < win_probability:
                # Winning trade
                trade['result'] = 'WIN'
                trade['exit_price'] = trade['tp_price']
                if trade['signal'] == 'BUY':
                    trade['profit_loss'] = trade['tp_price'] - trade['entry_price']
                else:
                    trade['profit_loss'] = trade['entry_price'] - trade['tp_price']
            else:
                # Losing trade
                trade['result'] = 'LOSS'
                trade['exit_price'] = trade['sl_price']
                if trade['signal'] == 'BUY':
                    trade['profit_loss'] = trade['sl_price'] - trade['entry_price']
                else:
                    trade['profit_loss'] = trade['entry_price'] - trade['sl_price']
            
            trade['status'] = 'CLOSED'
            self._update_statistics()
            self._save_performance_data()
            
        except Exception as e:
            logger.error(f"Error simulating trade outcome: {str(e)}")
    
    def _update_statistics(self):
        """Update performance statistics"""
        try:
            trades = [t for t in self.performance_data['trades'] if t['status'] == 'CLOSED']
            
            if not trades:
                return
            
            stats = self.performance_data['statistics']
            
            # Basic counts
            stats['total_trades'] = len(trades)
            winning_trades = [t for t in trades if t['result'] == 'WIN']
            losing_trades = [t for t in trades if t['result'] == 'LOSS']
            
            stats['winning_trades'] = len(winning_trades)
            stats['losing_trades'] = len(losing_trades)
            
            # Win rate
            stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
            
            # Profit/Loss calculations
            profits = [t['profit_loss'] for t in winning_trades]
            losses = [abs(t['profit_loss']) for t in losing_trades]
            
            stats['total_profit'] = sum(profits) if profits else 0
            stats['total_loss'] = sum(losses) if losses else 0
            
            # Profit factor
            stats['profit_factor'] = (stats['total_profit'] / stats['total_loss']) if stats['total_loss'] > 0 else 0
            
            # Largest win/loss
            stats['largest_win'] = max(profits) if profits else 0
            stats['largest_loss'] = max(losses) if losses else 0
            
            # Average win/loss
            stats['avg_win'] = sum(profits) / len(profits) if profits else 0
            stats['avg_loss'] = sum(losses) / len(losses) if losses else 0
            
            # Streak calculations
            current_streak = 0
            best_streak = 0
            worst_streak = 0
            temp_streak = 0
            
            for trade in reversed(trades):
                if trade['result'] == 'WIN':
                    if temp_streak >= 0:
                        temp_streak += 1
                    else:
                        temp_streak = 1
                else:
                    if temp_streak <= 0:
                        temp_streak -= 1
                    else:
                        temp_streak = -1
                
                best_streak = max(best_streak, temp_streak)
                worst_streak = min(worst_streak, temp_streak)
            
            current_streak = temp_streak
            
            stats['current_streak'] = current_streak
            stats['best_streak'] = best_streak
            stats['worst_streak'] = worst_streak
            
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
    
    def get_performance_summary(self):
        """Get current performance summary"""
        try:
            stats = self.performance_data['statistics']
            recent_trades = self._get_recent_trades(limit=10)
            
            return {
                'summary': {
                    'total_trades': stats['total_trades'],
                    'win_rate': round(stats['win_rate'], 1),
                    'profit_factor': round(stats['profit_factor'], 2),
                    'current_streak': stats['current_streak'],
                    'largest_win': round(stats['largest_win'] * 10000, 1),  # Convert to pips
                    'largest_loss': round(stats['largest_loss'] * 10000, 1)
                },
                'streak_info': {
                    'current': stats['current_streak'],
                    'best': stats['best_streak'],
                    'worst': stats['worst_streak']
                },
                'recent_trades': recent_trades,
                'last_updated': self.performance_data.get('last_updated'),
                'system_age_days': self._get_system_age_days()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {
                'summary': {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'current_streak': 0},
                'streak_info': {'current': 0, 'best': 0, 'worst': 0},
                'recent_trades': [],
                'error': str(e)
            }
    
    def _get_recent_trades(self, limit=10):
        """Get recent closed trades"""
        try:
            closed_trades = [t for t in self.performance_data['trades'] if t['status'] == 'CLOSED']
            recent = closed_trades[-limit:] if closed_trades else []
            
            # Format for display
            formatted_trades = []
            for trade in recent:
                formatted_trades.append({
                    'pair': trade['pair'],
                    'signal': trade['signal'],
                    'result': trade['result'],
                    'profit_pips': round(trade['profit_loss'] * 10000, 1),
                    'confidence': trade['confidence'].get('percentage', 0),
                    'timestamp': trade['timestamp'][:10]  # Just date
                })
            
            return formatted_trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {str(e)}")
            return []
    
    def _get_system_age_days(self):
        """Calculate how many days the system has been running"""
        try:
            start_date_str = self.performance_data.get('system_start_date')
            if start_date_str:
                start_date = datetime.fromisoformat(start_date_str)
                age = (datetime.now() - start_date).days
                return max(1, age)  # At least 1 day
            return 1
        except:
            return 1