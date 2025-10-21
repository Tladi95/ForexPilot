import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexDataFetcher:
    """Fetches real-time forex data using yfinance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def get_forex_data(self, pair, period='1d', interval='1h'):
        """
        Fetch forex data for a given pair
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD=X', 'USDJPY=X')
            period (str): Data period ('1d', '5d', '1mo', etc.)
            interval (str): Data interval ('1m', '5m', '15m', '1h', etc.)
        
        Returns:
            pandas.DataFrame: OHLCV data with technical indicators
        """
        try:
            # Check cache first
            cache_key = f"{pair}_{period}_{interval}"
            current_time = datetime.now()
            
            if cache_key in self.cache:
                cached_data, cache_time = self.cache[cache_key]
                if (current_time - cache_time).seconds < self.cache_duration:
                    logger.info(f"Using cached data for {pair}")
                    return cached_data
            
            # Fetch fresh data
            logger.info(f"Fetching fresh data for {pair}")
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned for {pair}")
                return None
            
            # Clean and prepare data
            data = data.dropna()
            
            # Add basic price data if needed
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
            
            # Cache the data
            self.cache[cache_key] = (data, current_time)
            
            logger.info(f"Successfully fetched {len(data)} rows for {pair}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {str(e)}")
            return None
    
    def get_multi_timeframe_data(self, pair):
        """
        Fetch multi-timeframe data for comprehensive analysis
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD=X', 'USDJPY=X')
        
        Returns:
            dict: Data for different timeframes
        """
        try:
            # Intraday-focused timeframes for institutional analysis (5m-15m optimized)
            timeframes = {
                '5M': ('7d', '5m'),     # 5-minute data for 7 days (high frequency)
                '15M': ('30d', '15m'),  # 15-minute data for 30 days (primary timeframe)
                '1H': ('60d', '1h'),    # 1-hour data for trend confirmation
            }
            
            multi_data = {}
            
            for tf_name, (period, interval) in timeframes.items():
                try:
                    data = self.get_forex_data(pair, period=period, interval=interval)
                    if data is not None and not data.empty:
                        multi_data[tf_name] = data
                    else:
                        multi_data[tf_name] = None
                        logger.warning(f"No {tf_name} data for {pair}")
                except Exception as e:
                    logger.warning(f"Error fetching {tf_name} data for {pair}: {str(e)}")
                    multi_data[tf_name] = None
            
            return multi_data
            
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe data for {pair}: {str(e)}")
            return {}
    
    def get_current_price(self, pair):
        """Get current price for a currency pair"""
        try:
            data = self.get_forex_data(pair, period='1d', interval='1m')
            if data is not None and not data.empty:
                return data['Close'].iloc[-1]
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {pair}: {str(e)}")
            return None