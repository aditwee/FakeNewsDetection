from flask import Flask, request, jsonify, render_template
from inference import predict_fake_news
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        if not data:
            # Try form data if JSON is not available
            title = request.form.get('title', '').strip()
            body = request.form.get('body', '').strip()
        else:
            title = data.get('title', '').strip()
            body = data.get('body', '').strip()
        
        # Validate input
        if not title and not body:
            return jsonify({
                'error': 'Please provide at least a title or body text'
            }), 400
        
        # Make prediction using the inference function
        result = predict_fake_news(title, body)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists('model.pt')
    })

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Model should be loaded from inference.py")
    app.run(host='0.0.0.0', port=5000, debug=True)

