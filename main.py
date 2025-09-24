from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import traceback
from trading.data_fetcher import ForexDataFetcher
from trading.ensemble_analyzer import EnsembleAnalyzer
from trading.risk_manager import RiskManager
from trading.performance_tracker import PerformanceTracker

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'forex-trading-secret-key')

# Initialize trading components
data_fetcher = ForexDataFetcher()
ensemble_analyzer = EnsembleAnalyzer()
risk_manager = RiskManager()
performance_tracker = PerformanceTracker()

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
        
        # Currency pairs to analyze
        pairs = ['EURUSD=X', 'USDJPY=X']
        results = {}
        
        for pair in pairs:
            # Fetch multi-timeframe data for enhanced analysis
            data_1h = data_fetcher.get_forex_data(pair, period='60d', interval='1h')
            multi_timeframe_data = data_fetcher.get_multi_timeframe_data(pair)
            
            if data_1h is not None and not data_1h.empty:
                # Enhanced ensemble analysis with multi-timeframe support
                analysis = ensemble_analyzer.analyze(data_1h, multi_timeframe_data)
                
                # Calculate ATR-based risk management levels
                current_price = data_1h['Close'].iloc[-1]
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
                    'risk_amount': risk_calculations['risk_amount'],
                    'calculation_method': risk_calculations['calculation_method']
                }
            else:
                pair_name = pair.replace('=X', '').replace('USD', '/USD')
                results[pair_name] = {
                    'error': 'Unable to fetch data for this pair'
                }
        
        # Get performance statistics
        performance_summary = performance_tracker.get_performance_summary()
        
        return jsonify({
            'success': True,
            'data': results,
            'performance': performance_summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Run with host 0.0.0.0 to allow external connections
    app.run(host='0.0.0.0', port=5000, debug=True)