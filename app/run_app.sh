#!/bin/bash

# Script to run CodeBuggy Web Application

echo "🐛 Starting CodeBuggy Web Application..."

# Check if MLflow is running
echo "Checking MLflow connection..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✓ MLflow is running at http://localhost:5000"
else
    echo "⚠️  Warning: MLflow is not running at http://localhost:5000"
    echo "   Please start MLflow first:"
    echo "   mlflow server --host 0.0.0.0 --port 5000"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

curl -LsSf https://astral.sh/uv/install.sh | sh

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12.10
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../.venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install pip
uv pip install -q -r app_requirements.txt

# Check if node_type mapping exists
if [ ! -f "resources/node_type_to_id.joblib" ]; then
    echo "⚠️  Warning: resources/node_type_to_id.joblib not found"
    echo "   This file is created during model training"
    echo "   The app will use a default mapping"
fi

# Set environment variables
export MLFLOW_URI=${MLFLOW_URI:-http://localhost:5000}
export MODEL_NAME=${MODEL_NAME:-codebuggy-detector}
export MODEL_STAGE=${MODEL_STAGE:-Production}
export PORT=${PORT:-8080}

echo ""
echo "Configuration:"
echo "  MLFLOW_URI: $MLFLOW_URI"
echo "  MODEL_NAME: $MODEL_NAME"
echo "  MODEL_STAGE: $MODEL_STAGE"
echo "  PORT: $PORT"
echo ""

# Run application
echo "Starting Flask application..."
echo "Access the app at: http://localhost:$PORT"
echo ""
python app.py
