from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import traceback
from trading.data_fetcher import ForexDataFetcher
from trading.ensemble_analyzer import EnsembleAnalyzer
from trading.risk_manager import RiskManager

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'forex-trading-secret-key')

# Initialize trading components
data_fetcher = ForexDataFetcher()
ensemble_analyzer = EnsembleAnalyzer()
risk_manager = RiskManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/signals')
def get_signals():
    """API endpoint to get trading signals for both currency pairs"""
    try:
        account_size = request.args.get('account_size', 10000, type=float)
        risk_percent = request.args.get('risk_percent', 1.5, type=float)
        
        # Currency pairs to analyze
        pairs = ['EURUSD=X', 'USDJPY=X']
        results = {}
        
        for pair in pairs:
            # Fetch real-time data (get more history for technical analysis)
            data = data_fetcher.get_forex_data(pair, period='60d', interval='1h')
            
            if data is not None and not data.empty:
                # Analyze with ensemble models
                analysis = ensemble_analyzer.analyze(data)
                
                # Calculate risk management levels
                current_price = data['Close'].iloc[-1]
                risk_calculations = risk_manager.calculate_risk_levels(
                    pair=pair.replace('=X', ''),
                    current_price=current_price,
                    account_size=account_size,
                    risk_percent=risk_percent
                )
                
                # Combine analysis and risk data
                pair_name = pair.replace('=X', '').replace('USD', '/USD')
                results[pair_name] = {
                    'timestamp': datetime.now().isoformat(),
                    'current_price': round(current_price, 5),
                    'signal': analysis['final_signal'],
                    'buy_probability': analysis['buy_probability'],
                    'sell_probability': analysis['sell_probability'],
                    'model_votes': analysis['model_votes'],
                    'stop_loss': risk_calculations['stop_loss'],
                    'take_profit': risk_calculations['take_profit'],
                    'position_size': risk_calculations['position_size'],
                    'risk_amount': risk_calculations['risk_amount']
                }
            else:
                pair_name = pair.replace('=X', '').replace('USD', '/USD')
                results[pair_name] = {
                    'error': 'Unable to fetch data for this pair'
                }
        
        return jsonify({
            'success': True,
            'data': results,
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