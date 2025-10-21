from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import traceback
import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def sanitize_for_json(obj):
    """Convert NumPy/Pandas objects to JSON-serializable types"""
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif hasattr(obj, 'item'):  # NumPy scalar
        return obj.item()
    elif pd.isna(obj):
        return None
    else:
        return obj
from trading.data_fetcher import ForexDataFetcher
from trading.ensemble_analyzer import EnsembleAnalyzer
from trading.risk_manager import RiskManager
from trading.performance_tracker import PerformanceTracker
from trading.backtester import InstitutionalBacktester

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET')

# Initialize trading components
data_fetcher = ForexDataFetcher()
ensemble_analyzer = EnsembleAnalyzer()
risk_manager = RiskManager()
performance_tracker = PerformanceTracker()
backtester = InstitutionalBacktester(ensemble_analyzer, risk_manager)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/signals')
def get_signals():
    """Enhanced API endpoint with multi-timeframe analysis and ATR-based calculations"""
    try:
        account_size = request.args.get('account_size', 10000, type=float)
        risk_percent = request.args.get('risk_percent', 1.5, type=float)
        
        # Currency pairs to analyze (expanded for more trading opportunities)
        pairs = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X', 'AUDUSD=X', 'NZDUSD=X']
        results = {}
        
        for pair in pairs:
            # Fetch intraday data optimized for 5m-15m institutional analysis
            data_15m = data_fetcher.get_forex_data(pair, period='30d', interval='15m')
            multi_timeframe_data = data_fetcher.get_multi_timeframe_data(pair)
            
            if data_15m is not None and not data_15m.empty:
                # Enhanced ensemble analysis with intraday 15m data
                analysis = ensemble_analyzer.analyze(data_15m, multi_timeframe_data)
                
                # Calculate institutional-grade risk management levels
                current_price = data_15m['Close'].iloc[-1]
                
                # Extract Kelly fraction and institutional features from analysis (with safe defaults)
                kelly_fraction = analysis.get('kelly_fraction', None)
                timeframe = '15m'  # Default intraday timeframe
                
                # Calculate risk levels with error handling for institutional features
                try:
                    risk_calculations = risk_manager.calculate_risk_levels(
                        pair=pair.replace('=X', ''),
                        current_price=current_price,
                        account_size=account_size,
                        risk_percent=risk_percent,
                        atr_data=analysis.get('atr_value'),
                        timeframe=timeframe,
                        kelly_fraction=kelly_fraction
                    )
                except Exception as risk_error:
                    logger.warning(f"Institutional risk calculation failed, using basic method: {str(risk_error)}")
                    # Fallback to basic risk calculation
                    risk_calculations = risk_manager.calculate_risk_levels(
                        pair=pair.replace('=X', ''),
                        current_price=current_price,
                        account_size=account_size,
                        risk_percent=risk_percent,
                        atr_data=analysis.get('atr_value')
                    )
                
                # Record signal for performance tracking
                if analysis['final_signal'] in ['BUY', 'SELL']:
                    performance_tracker.record_signal(
                        pair=pair.replace('=X', ''),
                        signal=analysis['final_signal'],
                        confidence=analysis['confidence'],
                        entry_price=current_price,
                        sl_price=risk_calculations['stop_loss']['buy' if analysis['final_signal'] == 'BUY' else 'sell'],
                        tp_price=risk_calculations['take_profit']['buy' if analysis['final_signal'] == 'BUY' else 'sell']
                    )
                
                # Combine analysis and risk data with enhanced information
                pair_name = pair.replace('=X', '').replace('USD', '/USD')
                results[pair_name] = {
                    'timestamp': datetime.now().isoformat(),
                    'current_price': round(current_price, 5),
                    'signal': analysis['final_signal'],
                    'confidence': analysis['confidence'],
                    'buy_probability': analysis['buy_probability'],
                    'sell_probability': analysis['sell_probability'],
                    'model_votes': analysis['model_votes'],
                    'market_regime': analysis['market_regime'],
                    'atr_value': analysis['atr_value'],
                    'price_action': analysis['price_action'],
                    'timeframe_consensus': analysis['timeframe_consensus'],
                    'stop_loss': risk_calculations['stop_loss'],
                    'take_profit': risk_calculations['take_profit'],
                    'position_size': risk_calculations['position_size'],
                    'risk_metrics': risk_calculations.get('risk_metrics', {'risk_amount': 0, 'risk_percent': 0}),
                    'session_limits': risk_calculations.get('session_limits', {}),
                    'reward_risk_ratio': risk_calculations.get('reward_risk_ratio', 2.0),
                    'timeframe': risk_calculations.get('timeframe', '15m'),
                    'calculation_method': risk_calculations.get('calculation_method', 'Basic'),
                    'institutional_features': risk_calculations.get('institutional_features', {}),
                    # Legacy compatibility
                    'risk_amount': risk_calculations.get('risk_metrics', {}).get('risk_amount', 
                                                       risk_calculations.get('risk_amount', 0))
                }
            else:
                pair_name = pair.replace('=X', '').replace('USD', '/USD')
                results[pair_name] = {
                    'error': 'Unable to fetch data for this pair'
                }
        
        # Get performance statistics
        performance_summary = performance_tracker.get_performance_summary()
        
        # Sanitize results for JSON serialization
        response_data = {
            'success': True,
            'data': results,
            'performance': performance_summary,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(sanitize_for_json(response_data))
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/signal')
def get_institutional_signal():
    """Enhanced institutional signal endpoint with EV, Kelly sizing, and comprehensive risk metrics"""
    try:
        account_size = request.args.get('account_size', 10000, type=float)
        risk_percent = request.args.get('risk_percent', 0.5, type=float)  # Conservative for intraday
        pair = request.args.get('pair', 'EURUSD', type=str)
        timeframe = request.args.get('timeframe', '15m', type=str)
        
        # Fetch intraday data
        if timeframe == '5m':
            data = data_fetcher.get_forex_data(f'{pair}=X', period='10d', interval='5m')
        elif timeframe == '15m':
            data = data_fetcher.get_forex_data(f'{pair}=X', period='30d', interval='15m')
        else:
            data = data_fetcher.get_forex_data(f'{pair}=X', period='60d', interval='1h')
        
        if data is None or data.empty:
            return jsonify({
                'success': False,
                'error': f'Unable to fetch data for {pair}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Get institutional analysis
        analysis = ensemble_analyzer.analyze(data)
        current_price = data['Close'].iloc[-1]
        
        # Calculate institutional risk levels
        risk_calculations = risk_manager.calculate_risk_levels(
            pair=pair,
            current_price=current_price,
            account_size=account_size,
            risk_percent=risk_percent,
            atr_data=analysis.get('atr_value'),
            timeframe=timeframe,
            kelly_fraction=analysis.get('kelly_fraction')
        )
        
        # Get risk summary
        risk_summary = risk_manager.get_risk_summary(account_size)
        
        # Prepare institutional response data
        institutional_response = {
            'success': True,
            'signal_data': {
                'pair': pair,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'current_price': round(current_price, 5),
                'signal': analysis['final_signal'],
                'confidence': analysis['confidence'],
                'institutional_metrics': {
                    'weighted_signal': analysis.get('weighted_signal', 0),
                    'calibrated_probability': analysis.get('calibrated_probability', 0.5),
                    'expected_value': analysis.get('expected_value', 0),
                    'kelly_fraction': analysis.get('kelly_fraction', 0),
                    'dynamic_threshold': analysis.get('dynamic_threshold', 0.6),
                    'meta_decision': analysis.get('meta_decision', 1)
                },
                'model_performance': {
                    'model_votes': analysis['model_votes'],
                    'model_weights': analysis.get('model_weights', {}),
                    'ensemble_details': analysis.get('model_details', {})
                },
                'market_analysis': {
                    'regime': analysis['market_regime'],
                    'price_action': analysis['price_action'],
                    'timeframe_consensus': analysis['timeframe_consensus']
                },
                'risk_management': risk_calculations,
                'session_summary': risk_summary
            },
            'filters_status': {
                'probability_filter': analysis.get('calibrated_probability', 0.5) >= analysis.get('dynamic_threshold', 0.6),
                'ev_filter': analysis.get('expected_value', 0) > 0,
                'meta_filter': analysis.get('meta_decision', 1) == 1,
                'all_passed': (analysis.get('calibrated_probability', 0.5) >= analysis.get('dynamic_threshold', 0.6) and 
                              analysis.get('expected_value', 0) > 0 and 
                              analysis.get('meta_decision', 1) == 1)
            }
        }
        
        # Sanitize and return response
        return jsonify(sanitize_for_json(institutional_response))
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/backtest')
def run_backtest():
    """Run institutional-grade backtest with comprehensive analytics"""
    try:
        pair = request.args.get('pair', 'EURUSD', type=str)
        timeframe = request.args.get('timeframe', '15m', type=str)
        account_size = request.args.get('account_size', 10000, type=float)
        
        # Fetch sufficient data for backtesting
        period_map = {'5m': '90d', '15m': '180d', '1h': '365d'}
        interval_map = {'5m': '5m', '15m': '15m', '1h': '1h'}
        
        data = data_fetcher.get_forex_data(
            f'{pair}=X', 
            period=period_map.get(timeframe, '180d'), 
            interval=interval_map.get(timeframe, '15m')
        )
        
        if data is None or data.empty or len(data) < 1000:
            return jsonify({
                'success': False,
                'error': f'Insufficient data for backtesting {pair} on {timeframe}',
                'data_points': len(data) if data is not None else 0
            }), 400
        
        # Run comprehensive backtest
        backtest_results = backtester.run_walk_forward_backtest(
            data=data,
            pair=pair,
            timeframe=timeframe,
            account_size=account_size
        )
        
        # Get backtest summary for comparison
        backtest_summary = backtester.get_backtest_summary(pair, timeframe)
        
        return jsonify({
            'success': True,
            'backtest_results': backtest_results,
            'historical_summary': backtest_summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/backtest')
def backtest_dashboard():
    """Institutional backtest analytics dashboard"""
    return render_template('backtest_dashboard.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',  # Updated version for institutional features
        'institutional_features': {
            'sharpe_weighted_ensemble': True,
            'probability_calibration': True,
            'dynamic_thresholds': True,
            'expected_value_filter': True,
            'kelly_sizing': True,
            'meta_labeling': True,
            'var_cvar_monitoring': True,
            'walk_forward_backtesting': True,
            'triple_barrier_labeling': True
        }
    })

if __name__ == '__main__':
    # Run with host 0.0.0.0 to allow external connections
    app.run(host='0.0.0.0', port=5000, debug=True)