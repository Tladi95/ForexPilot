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