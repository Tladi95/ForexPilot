import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskManager:
    """Risk Management System with TP/SL calculations"""
    
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
        
        # Stop loss pips for different pairs
        self.stop_loss_pips = {
            'EURUSD': 20,
            'USDJPY': 25,
            'GBPUSD': 25,
            'AUDUSD': 22,
            'USDCAD': 22,
            'USDCHF': 20
        }
        
        # Take profit pips (2:1 ratio for EUR/USD, 2:1 for USD/JPY)
        self.take_profit_pips = {
            'EURUSD': 40,  # 2:1 ratio
            'USDJPY': 50,  # 2:1 ratio
            'GBPUSD': 50,
            'AUDUSD': 44,
            'USDCAD': 44,
            'USDCHF': 40
        }
    
    def calculate_risk_levels(self, pair, current_price, account_size=10000.0, risk_percent=1.5):
        """
        Calculate stop loss, take profit, and position size
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            current_price (float): Current market price
            account_size (float): Account balance in USD
            risk_percent (float): Risk percentage per trade (1-2%)
        
        Returns:
            dict: Risk management calculations
        """
        try:
            # Get risk parameters for the pair
            sl_pips = self.stop_loss_pips.get(pair, 20)
            tp_pips = self.take_profit_pips.get(pair, 40)
            pip_value = self.pip_values.get(pair, 10.0)
            
            # Calculate pip size based on currency pair
            if 'JPY' in pair:
                pip_size = 0.01  # For JPY pairs, 1 pip = 0.01
            else:
                pip_size = 0.0001  # For other pairs, 1 pip = 0.0001
            
            # Calculate risk amount
            risk_amount = account_size * (risk_percent / 100)
            
            # Calculate position size based on risk
            # Position Size = Risk Amount / (Stop Loss Pips * Pip Value * Lot Size)
            # For micro lots (0.01), we need to adjust the calculation
            
            risk_per_pip = sl_pips * pip_value * 0.01  # For micro lots
            position_size_lots = risk_amount / (sl_pips * pip_value)
            
            # Convert to micro lots for practical trading
            micro_lots = position_size_lots * 100  # 1 standard lot = 100 micro lots
            micro_lots = max(0.01, round(micro_lots, 2))  # Minimum 0.01 micro lots
            
            # Calculate actual risk with position size
            actual_risk = (micro_lots / 100) * sl_pips * pip_value
            
            # Calculate stop loss and take profit prices
            if pair.startswith('USD'):
                # For USD/XXX pairs (like USD/JPY)
                stop_loss_buy = current_price - (sl_pips * pip_size)
                stop_loss_sell = current_price + (sl_pips * pip_size)
                take_profit_buy = current_price + (tp_pips * pip_size)
                take_profit_sell = current_price - (tp_pips * pip_size)
            else:
                # For XXX/USD pairs (like EUR/USD)
                stop_loss_buy = current_price - (sl_pips * pip_size)
                stop_loss_sell = current_price + (sl_pips * pip_size)
                take_profit_buy = current_price + (tp_pips * pip_size)
                take_profit_sell = current_price - (tp_pips * pip_size)
            
            return {
                'stop_loss': {
                    'buy': round(stop_loss_buy, 5),
                    'sell': round(stop_loss_sell, 5),
                    'pips': sl_pips
                },
                'take_profit': {
                    'buy': round(take_profit_buy, 5),
                    'sell': round(take_profit_sell, 5),
                    'pips': tp_pips
                },
                'position_size': {
                    'micro_lots': micro_lots,
                    'standard_lots': round(position_size_lots, 4),
                    'units': int(micro_lots * 1000)  # Micro lot = 1000 units
                },
                'risk_amount': round(actual_risk, 2),
                'risk_percent': round((actual_risk / account_size) * 100, 2),
                'reward_risk_ratio': round(tp_pips / sl_pips, 2),
                'pip_value': pip_value,
                'pip_size': pip_size
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk levels for {pair}: {str(e)}")
            return self._default_risk_calculation(current_price)
    
    def _default_risk_calculation(self, current_price):
        """Return default risk calculation when error occurs"""
        return {
            'stop_loss': {'buy': current_price * 0.998, 'sell': current_price * 1.002, 'pips': 20},
            'take_profit': {'buy': current_price * 1.004, 'sell': current_price * 0.996, 'pips': 40},
            'position_size': {'micro_lots': 0.01, 'standard_lots': 0.0001, 'units': 10},
            'risk_amount': 0,
            'risk_percent': 0,
            'reward_risk_ratio': 2.0,
            'pip_value': 10.0,
            'pip_size': 0.0001
        }