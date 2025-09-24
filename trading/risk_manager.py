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
    
    def calculate_risk_levels(self, pair, current_price, account_size=10000.0, risk_percent=1.5, atr_data=None):
        """
        Calculate stop loss, take profit, and position size using ATR or fallback to fixed pips
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            current_price (float): Current market price
            account_size (float): Account balance in USD
            risk_percent (float): Risk percentage per trade (1-2%)
            atr_data (dict): ATR data from ensemble analysis (optional)
        
        Returns:
            dict: Enhanced risk management calculations
        """
        try:
            # Calculate pip size based on currency pair
            if 'JPY' in pair:
                pip_size = 0.01  # For JPY pairs, 1 pip = 0.01
            else:
                pip_size = 0.0001  # For other pairs, 1 pip = 0.0001
            
            # Use ATR-based calculations if available, otherwise fallback to fixed pips
            if atr_data and atr_data.get('value', 0) > 0:
                # ATR-based dynamic levels
                atr_value = atr_data['value']
                sl_distance = atr_value * 1.5  # 1.5x ATR for stop loss
                tp_distance = atr_value * 3.0  # 3.0x ATR for take profit
                
                # Convert ATR distances to pips for position sizing
                sl_pips = sl_distance / pip_size
                tp_pips = tp_distance / pip_size
                
                calculation_method = 'ATR-based'
            else:
                # Fallback to fixed pip calculations
                sl_pips = self.stop_loss_pips.get(pair, 20)
                tp_pips = self.take_profit_pips.get(pair, 40)
                sl_distance = sl_pips * pip_size
                tp_distance = tp_pips * pip_size
                
                calculation_method = 'Fixed pips'
            
            # Get pip value for position sizing
            pip_value = self.pip_values.get(pair, 10.0)
            
            # Calculate risk amount
            risk_amount = account_size * (risk_percent / 100)
            
            # Calculate position size based on risk
            position_size_lots = risk_amount / (sl_pips * pip_value)
            
            # Convert to micro lots for practical trading
            micro_lots = position_size_lots * 100  # 1 standard lot = 100 micro lots
            micro_lots = max(0.01, round(micro_lots, 2))  # Minimum 0.01 micro lots
            
            # Calculate actual risk with position size
            actual_risk = (micro_lots / 100) * sl_pips * pip_value
            
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
                    'units': int(micro_lots * 1000)  # Micro lot = 1000 units
                },
                'risk_amount': round(actual_risk, 2),
                'risk_percent': round((actual_risk / account_size) * 100, 2),
                'reward_risk_ratio': round(tp_pips / sl_pips, 2) if sl_pips > 0 else 2.0,
                'pip_value': pip_value,
                'pip_size': pip_size,
                'calculation_method': calculation_method,
                'atr_used': atr_data is not None and atr_data.get('value', 0) > 0
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