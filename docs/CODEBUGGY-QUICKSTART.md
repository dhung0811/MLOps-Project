# CodeBuggy MLflow Quick Start

## Tổng Quan

Hướng dẫn nhanh để log và sử dụng CodeBuggy RGCN model với MLflow.

## Prerequisites

```bash
# Ensure MLflow is running
kubectl port-forward svc/mlflow 5000:5000 -n mlops

# Install dependencies
pip3 install torch torch-geometric transformers mlflow javalang
```

## Bước 1: Log Model vào MLflow

```bash
# Log trained model to MLflow Registry
python3 model/codebuggy_train_mlflow.py \
  --checkpoint output/rgcn_detector.pt \
  --node-types output/node_type_to_id.joblib \
  --mlflow-uri http://localhost:5000
```

**Output:**
```
Loading checkpoint from output/rgcn_detector.pt...
Loaded node type mapping: 87 types

Model parameters:
  base_in_dim: 774
  hidden_dim: 256
  num_relations: 10
  num_node_types: 87

Recreating model...
  Model loaded successfully

Logging model to MLflow...

================================================================================
✓ Model logged to MLflow!
================================================================================
Run ID: abc123...
Model registered as: codebuggy-detector
```

## Bước 2: Promote Model to Production

```bash
python3 -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()

# Get latest version
versions = client.search_model_versions(\"name='codebuggy-detector'\")
latest = max(versions, key=lambda x: int(x.version))

# Promote to Production
client.transition_model_version_stage(
    name='codebuggy-detector',
    version=latest.version,
    stage='Production'
)

print(f'✓ Model v{latest.version} promoted to Production')
"
```

## Bước 3: Test Inference

### Option A: Sử dụng Example Script

```bash
python3 model/codebuggy_example.py
```

Sẽ chạy 3 ví dụ:
1. Array Index Out of Bounds
2. Null Pointer Dereference  
3. Resource Leak

### Option B: Inference với Custom Code

```bash
python3 model/codebuggy_infer_complete.py \
  --buggy "public int sum(int[] arr) { int s = 0; for (int i = 0; i <= arr.length; i++) { s += arr[i]; } return s; }" \
  --fixed "public int sum(int[] arr) { int s = 0; for (int i = 0; i < arr.length; i++) { s += arr[i]; } return s; }" \
  --mlflow-uri http://localhost:5000 \
  --model-name codebuggy-detector \
  --model-stage Production
```

### Option C: Inference từ Files

```bash
# Tạo test files
cat > /tmp/buggy.java << 'EOF'
public int sum(int[] arr) {
    int s = 0;
    for (int i = 0; i <= arr.length; i++) {
        s += arr[i];
    }
    return s;
}
EOF

cat > /tmp/fixed.java << 'EOF'
public int sum(int[] arr) {
    int s = 0;
    for (int i = 0; i < arr.length; i++) {
        s += arr[i];
    }
    return s;
}
EOF

# Run inference
python3 model/codebuggy_infer_complete.py \
  --buggy-file /tmp/buggy.java \
  --fixed-file /tmp/fixed.java \
  --log-mlflow
```

## Bước 4: Xem Results trong MLflow UI

```bash
# Open MLflow UI
open http://localhost:5000
```

Navigate to:
- **Experiments** → `codebuggy-rgcn` → View training runs
- **Models** → `codebuggy-detector` → View versions
- **Runs** → View inference results (if logged)

## Python API Usage

```python
from model.codebuggy_infer_complete import CodeBuggyPredictor

# Initialize
predictor = CodeBuggyPredictor(
    mlflow_uri="http://localhost:5000",
    model_name="codebuggy-detector",
    model_stage="Production",
)

# Predict
buggy_code = """
public int sum(int[] arr) {
    int s = 0;
    for (int i = 0; i <= arr.length; i++) {
        s += arr[i];
    }
    return s;
}
"""

fixed_code = """
public int sum(int[] arr) {
    int s = 0;
    for (int i = 0; i < arr.length; i++) {
        s += arr[i];
    }
    return s;
}
"""

results = predictor.predict(buggy_code, fixed_code, log_to_mlflow=True)

print(f"Graph bug probability: {results['graph_probability']:.4f}")
print(f"Top buggy nodes: {results['node_probabilities'][:5]}")
```

## Troubleshooting

### Model not found
```bash
# Check if model is registered
python3 -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
models = client.search_registered_models()
for m in models:
    print(f'{m.name}: {len(m.latest_versions)} versions')
"
```

### Port-forward not working
```bash
# Check MLflow pod
kubectl get pods -n mlops

# Restart port-forward
kubectl port-forward svc/mlflow 5000:5000 -n mlops
```

### GumTree not available
GumTree is optional. The inference will use simplified diff features if GumTree is not available.

To enable GumTree:
```bash
# Download GumTree
wget https://github.com/GumTreeDiff/gumtree/releases/download/v4.0.0-beta4/gumtree-4.0.0-beta4.zip
unzip gumtree-4.0.0-beta4.zip

# Update path in predictor
predictor = CodeBuggyPredictor(
    gumtree_path="./gumtree-4.0.0-beta4/bin/gumtree"
)
```

## Next Steps

- [ ] Batch inference trên dataset
- [ ] Deploy API endpoint (Day 2)
- [ ] Setup drift detection
- [ ] Integrate với Kubeflow (Day 3)
