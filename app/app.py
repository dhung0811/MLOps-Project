"""
Flask Web Application for CodeBuggy Model Serving
Provides UI to input buggy and fixed code, displays predictions
"""
from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from inference import CodeBuggyPredictor

app = Flask(__name__)

# Global predictor instance
predictor = None

# Load examples
def load_examples():
    """Load example code snippets"""
    examples_path = os.path.join(os.path.dirname(__file__), 'examples.json')
    if os.path.exists(examples_path):
        with open(examples_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"examples": []}

def initialize_predictor():
    """Initialize the predictor on first request"""
    global predictor
    if predictor is None:
        mlflow_uri = os.getenv("MLFLOW_URI", "http://localhost:5000")
        model_name = os.getenv("MODEL_NAME", "codebuggy-detector")
        model_stage = os.getenv("MODEL_STAGE", "Production")
        
        predictor = CodeBuggyPredictor(
            mlflow_uri=mlflow_uri,
            model_name=model_name,
            model_stage=model_stage,
        )
    return predictor


@app.route('/')
def index():
    """Main page with input form"""
    examples = load_examples()
    return render_template('index.html', examples=examples['examples'])


@app.route('/api/examples')
def get_examples():
    """Get all examples as JSON"""
    examples = load_examples()
    return jsonify(examples)


@app.route('/examples')
def examples_page():
    """Examples gallery page"""
    examples = load_examples()
    return render_template('examples.html', examples=examples['examples'])


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = request.get_json()
        buggy_code = data.get('buggy_code', '').strip()
        fixed_code = data.get('fixed_code', '').strip()
        
        if not buggy_code or not fixed_code:
            return jsonify({
                'success': False,
                'error': 'Both buggy and fixed code are required'
            }), 400
        
        print(f"\n{'='*80}")
        print(f"Received prediction request")
        print(f"Buggy code length: {len(buggy_code)}")
        print(f"Fixed code length: {len(fixed_code)}")
        print(f"{'='*80}\n")
        
        # Initialize predictor
        pred = initialize_predictor()
        
        # Run prediction
        results = pred.predict(buggy_code, fixed_code, log_to_mlflow=False)
        
        # Format node predictions
        node_predictions = []
        node_probs = results['node_probabilities']
        nodes = results['nodes']
        
        # Get top 20 predictions
        ranked = sorted(range(len(node_probs)), key=lambda i: node_probs[i], reverse=True)
        
        for idx in ranked[:20]:
            node = nodes[idx]
            node_predictions.append({
                'probability': float(node_probs[idx]),
                'node_type': node.get('node_type'),
                'label': node.get('label') or '',
                'line': node.get('line'),
                'col': node.get('col'),
            })
        
        print(f"\nReturning {len(node_predictions)} predictions")
        print(f"Graph probability: {results['graph_probability']:.4f}\n")
        
        return jsonify({
            'success': True,
            'graph_probability': results['graph_probability'],
            'node_predictions': node_predictions,
            'num_nodes': results['num_nodes'],
            'num_edges': results['num_edges'],
        })
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n{'='*80}")
        print(f"Error during prediction: {error_msg}")
        print(f"{'='*80}")
        traceback.print_exc()
        print(f"{'='*80}\n")
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        pred = initialize_predictor()
        return jsonify({
            'status': 'healthy',
            'model_loaded': pred.model is not None
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
