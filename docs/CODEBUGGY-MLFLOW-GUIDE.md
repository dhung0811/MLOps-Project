# CodeBuggy + MLflow Integration Guide

## Overview

Tích hợp MLflow vào CodeBuggy RGCN model để:
- Track experiments và hyperparameters
- Version model trong Model Registry
- Log inference results
- Reproduce experiments

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CODEBUGGY + MLFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Training   │─────▶│   MLflow     │◀─────│  Inference   │  │
│  │              │      │   Server     │      │              │  │
│  │ • Train RGCN │      │              │      │ • Load model │  │
│  │ • Log params │      │ • Tracking   │      │ • Predict    │  │
│  │ • Log metrics│      │ • Registry   │      │ • Log results│  │
│  │ • Save model │      │ • Artifacts  │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created

```
model/
├── codebuggy_train_mlflow.py    # Log trained model to MLflow
└── codebuggy_infer_mlflow.py    # Load model from MLflow & inference
```

## Step 1: Log Trained Model to MLflow

### Prepare

Đảm bảo bạn đã có:
- Trained checkpoint: `output/rgcn_detector.pt`
- Node type mapping: `output/node_type_to_id.joblib`
- Training history: `output/train_hist.pt` (optional)

### Code

```python
# model/codebuggy_train_mlflow.py
from codebuggy_train_mlflow import log_model_to_mlflow
from your_module import RGCNDetector  # Import model class

log_model_to_mlflow(
    checkpoint_path="output/rgcn_detector.pt",
    model_class=RGCNDetector,
    model_params={
        "num_node_types": 100,  # Số lượng node types
        "num_layers": 2,
        "dropout": 0.2,
        "node_type_emb_dim": 64,
    },
    training_history=torch.load("output/train_hist.pt"),
)
```

### What Gets Logged

| Artifact | Description |
|----------|-------------|
| **Parameters** | base_in_dim, hidden_dim, num_relations, dropout, etc. |
| **Metrics** | train_loss, val_loss, node_f1, graph_f1 per epoch |
| **Model** | PyTorch model với state_dict |
| **Artifacts** | checkpoint.pt, node_type_to_id.joblib, train_hist.pt |

### Run

```bash
# Ensure MLflow port-forward is running
kubectl port-forward svc/mlflow 5000:5000 -n mlops

# Log model
python model/codebuggy_train_mlflow.py --checkpoint output/rgcn_detector.pt
```

## Step 2: Promote Model to Production

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

# Get latest version
versions = client.search_model_versions("name='codebuggy-detector'")
latest = max(versions, key=lambda x: int(x.version))

# Promote to Production
client.transition_model_version_stage(
    name="codebuggy-detector",
    version=latest.version,
    stage="Production"
)

print(f"Model v{latest.version} promoted to Production")
```

## Step 3: Inference with MLflow

### Load Model

```python
from codebuggy_infer_mlflow import CodeBuggyInference

# Initialize
inferencer = CodeBuggyInference(mlflow_uri="http://localhost:5000")

# Load model from registry
inferencer.load_model_from_mlflow(
    model_name="codebuggy-detector",
    stage="Production"
)
```

### Predict

```python
# Build graph từ code (sử dụng code từ notebook)
from your_notebook import build_graph_sample

data, parts = build_graph_sample(buggy_code, fixed_code, "test_1")

# Predict
node_probs, graph_prob = inferencer.predict(data)

print(f"Graph bug probability: {graph_prob:.4f}")
```

### Log Results

```python
# Log inference results to MLflow
inferencer.log_inference_results(
    buggy_code=buggy_code,
    fixed_code=fixed_code,
    node_probs=node_probs,
    graph_prob=graph_prob,
    nodes=parts["nodes"],
    top_k=10
)
```

## Step 4: Batch Inference

```python
import pandas as pd

# Load dataset
df = pd.read_csv('megadiff_single_function_100.csv')

for idx in range(len(df)):
    buggy_code = df['buggy_function'].iloc[idx]
    fixed_code = df['fixed_function'].iloc[idx]
    
    # Build graph
    data, parts = build_graph_sample(buggy_code, fixed_code, f"sample_{idx}")
    
    # Predict
    node_probs, graph_prob = inferencer.predict(data)
    
    # Log to MLflow
    inferencer.log_inference_results(
        buggy_code, fixed_code, node_probs, graph_prob, parts["nodes"]
    )
    
    print(f"Processed {idx+1}/{len(df)}")
```

## MLflow UI

Access MLflow UI để xem:
- **Experiments**: Training runs với metrics/params
- **Models**: Registered models với versions
- **Artifacts**: Checkpoints, code, predictions

```bash
# Port-forward
kubectl port-forward svc/mlflow 5000:5000 -n mlops

# Open browser
open http://localhost:5000
```

## Integration với Notebook

Để sử dụng code từ notebook `codebuggy-infer.ipynb`:

### Option 1: Convert notebook to module

```bash
# Extract functions từ notebook
jupyter nbconvert --to script codebuggy-infer.ipynb
# Edit codebuggy-infer.py để export functions
```

### Option 2: Import trực tiếp

```python
# Trong codebuggy_infer_mlflow.py
import sys
sys.path.append('path/to/notebook/dir')

from codebuggy_infer import (
    build_graph_sample,
    compute_code_embeddings,
    # ... other functions
)
```

### Option 3: Copy functions

Copy các functions cần thiết từ notebook vào module riêng:
- `build_ast_graph()`
- `build_graph_sample()`
- `compute_code_embeddings()`
- `match_actions_to_nodes()`

## Benefits

✅ **Reproducibility**: Track tất cả experiments
✅ **Versioning**: Quản lý model versions
✅ **Comparison**: So sánh models dễ dàng
✅ **Deployment**: Load model từ registry
✅ **Monitoring**: Track inference results

## Next Steps

1. [ ] Log trained model to MLflow
2. [ ] Promote model to Production
3. [ ] Test inference với MLflow model
4. [ ] Setup batch inference pipeline
5. [ ] Integrate với Kubeflow (Day 3)
