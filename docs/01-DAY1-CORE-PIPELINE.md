# NGÀY 1: MLFLOW SETUP & INTEGRATION

## Mục Tiêu
- Setup MLflow server
- Training script với MLflow logging
- CI pipeline validation

> **Note**: Docker image đã có sẵn, không cần build mới

## Task 1.1: MLflow Setup & Training (2 giờ)

### MLflow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLFLOW COMPONENTS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   TRACKING      │  │    MODEL        │  │   ARTIFACT      │ │
│  │   SERVER        │  │   REGISTRY      │  │    STORE        │ │
│  │                 │  │                 │  │                 │ │
│  │ • Experiments   │  │ • Model Versions│  │ • model.pkl     │ │
│  │ • Runs          │  │ • Stage (None/  │  │ • metrics       │ │
│  │ • Parameters    │  │   Staging/Prod) │  │ • artifacts     │ │
│  │ • Metrics       │  │ • Aliases       │  │                 │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │          │
│           └────────────────────┴────────────────────┘          │
│                              │                                  │
│                    ┌─────────▼─────────┐                       │
│                    │   Backend Store   │                       │
│                    │   (SQLite/MySQL)  │                       │
│                    └───────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Flow với MLflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Dataset   │────▶│   Train     │────▶│   Log to    │────▶│  Register   │
│  (Iris/CSV) │     │  (sklearn)  │     │   MLflow    │     │   Model     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │  Artifacts  │
                                        │ • model.pkl │
                                        │ • metrics   │
                                        │ • ref_data  │
                                        └─────────────┘
```

### INCLUDED
- MLflow experiment tracking
- Model versioning
- Artifact logging (model, reference data)
- Basic metrics logging

### SKIPPED
- MLflow server với database backend (dùng local file store)
- Model signatures
- Input examples


### Training Script với MLflow

```python
# model/train.py
import mlflow
import mlflow.sklearn
import pickle
import argparse
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Hoặc K8s service URL
EXPERIMENT_NAME = "iris-classification"
MODEL_NAME = "iris-model"


def train_model(register: bool = True):
    """Train model và log to MLflow"""
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted')
        }
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            registered_model_name=MODEL_NAME if register else None
        )
        
        # Log reference data for drift detection
        np.save("/tmp/reference_data.npy", X_train)
        mlflow.log_artifact("/tmp/reference_data.npy", artifact_path="drift")
        
        # Print results
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Metrics: {metrics}")
        
        if register:
            print(f"Model registered as: {MODEL_NAME}")
        
        return mlflow.active_run().info.run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", default=True)
    parser.add_argument("--mlflow-uri", default=MLFLOW_TRACKING_URI)
    args = parser.parse_args()
    
    MLFLOW_TRACKING_URI = args.mlflow_uri
    train_model(register=args.register)
```

---

## Task 1.2: MLflow Model Loading (30 phút)

### Load Model từ Registry

```python
# api/model_loader.py
import mlflow
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")  # None, Staging, Production


def load_model_from_registry():
    """Load model từ MLflow Registry"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load by stage
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model: {MODEL_NAME} @ {MODEL_STAGE}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to latest version
        model_uri = f"models:/{MODEL_NAME}/latest"
        return mlflow.sklearn.load_model(model_uri)


def load_reference_data(run_id: str = None):
    """Load reference data for drift detection"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    if run_id:
        artifact_uri = f"runs:/{run_id}/drift/reference_data.npy"
    else:
        # Get latest run
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
        artifact_uri = f"runs:/{model_version.run_id}/drift/reference_data.npy"
    
    local_path = mlflow.artifacts.download_artifacts(artifact_uri)
    return local_path


def get_model_info():
    """Get current model information"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            v = versions[0]
            return {
                "name": MODEL_NAME,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status
            }
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "No model found"}
```


---

## Task 1.3: CI Pipeline (1 giờ)

> **Note**: Bỏ qua Docker build vì đã có image sẵn

### CI Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CI PIPELINE (GitHub Actions)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐   ┌─────────────┐   ┌─────────────────────────┐   │
│  │  Lint   │──▶│ Model Load  │──▶│  Register to MLflow     │   │
│  │(flake8) │   │    Test     │   │  (if main branch)       │   │
│  └─────────┘   └─────────────┘   └─────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### GitHub Actions YAML

```yaml
# .github/workflows/ci.yaml
name: ML CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  MLFLOW_TRACKING_URI: http://localhost:5000

jobs:
  validate:
    runs-on: ubuntu-latest
    
    services:
      mlflow:
        image: ghcr.io/mlflow/mlflow:v2.9.2
        ports:
          - 5000:5000
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install mlflow==2.9.2 scikit-learn==1.3.2 numpy flake8
      
      - name: Lint
        run: flake8 model/ --max-line-length=120 --ignore=E501,W503
      
      - name: Train and register model
        run: |
          python model/train.py --mlflow-uri $MLFLOW_TRACKING_URI --register
      
      - name: Verify model in registry
        run: |
          python -c "
          import mlflow
          mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
          client = mlflow.tracking.MlflowClient()
          versions = client.search_model_versions(\"name='iris-model'\")
          assert len(versions) > 0, 'Model not found in registry'
          print(f'Model registered: {len(versions)} version(s)')
          "
```

---

## Deliverables Ngày 1

- [ ] MLflow tracking hoạt động (local)
- [ ] Training script log metrics/model to MLflow
- [ ] Model registered trong MLflow Registry
- [ ] CI pipeline pass

## Commands Tham Khảo

```bash
# Start MLflow server (local)
mlflow server --host 0.0.0.0 --port 5000

# Train model
python model/train.py --mlflow-uri http://localhost:5000

# View MLflow UI
open http://localhost:5000

# Verify model in registry
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
for v in client.search_model_versions(\"name='iris-model'\"):
    print(f'Version {v.version}: {v.current_stage}')
"
```
