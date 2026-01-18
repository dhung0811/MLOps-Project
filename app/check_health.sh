#!/bin/bash

# Health check script for CodeBuggy system

echo "🏥 CodeBuggy System Health Check"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. Check Python
echo "1. Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    check_pass "Python installed: $PYTHON_VERSION"
else
    check_fail "Python not found"
    exit 1
fi
echo ""

# 2. Check dependencies
echo "2. Checking Python dependencies..."
DEPS=("torch" "mlflow" "transformers" "javalang" "flask")
for dep in "${DEPS[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        VERSION=$(python3 -c "import $dep; print($dep.__version__)" 2>/dev/null || echo "unknown")
        check_pass "$dep ($VERSION)"
    else
        check_fail "$dep not installed"
    fi
done
echo ""

# 3. Check MLflow server
echo "3. Checking MLflow server..."
MLFLOW_URI=${MLFLOW_URI:-http://localhost:5000}
if curl -s "$MLFLOW_URI/health" > /dev/null 2>&1; then
    check_pass "MLflow server running at $MLFLOW_URI"
else
    check_fail "MLflow server not accessible at $MLFLOW_URI"
    echo "   Start with: mlflow server --host 0.0.0.0 --port 5000"
fi
echo ""

# 4. Check model in MLflow
echo "4. Checking model registration..."
MODEL_NAME=${MODEL_NAME:-codebuggy-detector}
if curl -s "$MLFLOW_URI/api/2.0/mlflow/registered-models/get?name=$MODEL_NAME" | grep -q "name"; then
    check_pass "Model '$MODEL_NAME' found in registry"
    
    # Check for Production stage
    if curl -s "$MLFLOW_URI/api/2.0/mlflow/registered-models/get?name=$MODEL_NAME" | grep -q "Production"; then
        check_pass "Production version exists"
    else
        check_warn "No Production version found (will use latest)"
    fi
else
    check_fail "Model '$MODEL_NAME' not found in registry"
    echo "   Train model with: python codebuggy_train_mlflow.py"
fi
echo ""

# 5. Check required files
echo "5. Checking required files..."
FILES=(
    "app.py"
    "codebuggy_infer_complete.py"
    "codebuggy_utils.py"
    "templates/index.html"
    "static/css/style.css"
    "static/js/app.js"
)
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file"
    else
        check_fail "$file not found"
    fi
done
echo ""

# 6. Check optional files
echo "6. Checking optional files..."
if [ -f "output/node_type_to_id.joblib" ]; then
    check_pass "node_type_to_id.joblib (from training)"
else
    check_warn "node_type_to_id.joblib not found (will use default)"
fi
echo ""

# 7. Check ports
echo "7. Checking ports..."
PORT=${PORT:-8080}
if lsof -i :$PORT > /dev/null 2>&1; then
    check_warn "Port $PORT is in use"
    echo "   Process: $(lsof -i :$PORT | tail -1)"
else
    check_pass "Port $PORT is available"
fi
echo ""

# 8. Check CUDA
echo "8. Checking CUDA availability..."
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    check_pass "CUDA available (version $CUDA_VERSION)"
else
    check_warn "CUDA not available (will use CPU)"
fi
echo ""

# Summary
echo "=================================="
echo "Health Check Complete"
echo ""
echo "To start the application:"
echo "  ./run_app.sh"
echo ""
echo "Or manually:"
echo "  python app.py"
echo ""
echo "Access at: http://localhost:${PORT}"
echo "=================================="
