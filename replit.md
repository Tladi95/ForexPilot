# Forex Trading System

## Overview

This is a Flask-based forex trading analysis system that provides real-time trading signals for EUR/USD and USD/JPY currency pairs. The system uses a 5-model ensemble approach to analyze market conditions and generate trading recommendations, combined with comprehensive risk management features. The application fetches live forex data, applies multiple technical analysis models, and calculates position sizing with stop-loss and take-profit levels.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Flask with Jinja2 templating
- **UI Framework**: Bootstrap 5.1.3 for responsive design
- **Icons**: Font Awesome 6.0.0 for visual elements
- **JavaScript**: Vanilla JavaScript for API interactions and dynamic updates
- **CSS**: Custom gradient-based styling with hover effects and responsive layouts

### Backend Architecture
- **Web Framework**: Flask application with modular trading components
- **API Design**: RESTful endpoints for signal generation and data retrieval
- **Session Management**: Flask session handling with configurable secret key
- **Error Handling**: Comprehensive exception handling with logging throughout the system

### Trading System Components
- **Data Fetcher**: Real-time forex data retrieval using yfinance library with caching mechanism
- **Ensemble Analyzer**: 5-model technical analysis system including:
  - SMA Crossover strategy
  - RSI analysis
  - MACD analysis  
  - Bollinger Bands strategy
  - Price Momentum analysis
- **Risk Manager**: Position sizing and risk calculation with configurable stop-loss and take-profit levels

### Data Processing
- **Technical Analysis**: Uses TA-Lib library for technical indicator calculations
- **Data Validation**: Input sanitization and data quality checks
- **Caching Strategy**: 5-minute cache duration for forex data to optimize API usage
- **Signal Aggregation**: Weighted voting system across multiple models for final recommendations

### Risk Management
- **Position Sizing**: Dynamic lot size calculation based on account size and risk percentage
- **Stop Loss/Take Profit**: Pair-specific pip calculations with 2:1 risk-reward ratios
- **Account Configuration**: User-configurable account size and risk tolerance settings

## External Dependencies

### Data Sources
- **yfinance**: Primary data source for real-time forex prices and historical data
- **Yahoo Finance API**: Underlying data provider for currency pair information

### Technical Analysis
- **TA-Lib**: Technical analysis library for indicator calculations
- **pandas**: Data manipulation and time series analysis
- **numpy**: Numerical computations and array operations

### Web Framework
- **Flask**: Core web application framework
- **Bootstrap CDN**: Frontend styling and responsive components
- **Font Awesome CDN**: Icon library for user interface elements

### Python Libraries
- **datetime**: Time-based operations and data timestamping
- **logging**: Application logging and error tracking
- **traceback**: Error diagnosis and debugging support

### Environment Configuration
- **SESSION_SECRET**: Environment variable for Flask session security
- **Default Configuration**: Fallback values for account size ($10,000) and risk percentage (1.5%)