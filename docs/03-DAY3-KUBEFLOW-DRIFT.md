# NGÀY 3: KUBEFLOW PIPELINES & DRIFT DETECTION

## Mục Tiêu
- Setup Kubeflow Pipelines
- Tạo retraining pipeline
- Drift detection trigger retrain

## Kubeflow Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KUBEFLOW PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRIGGER                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Manual (UI/API)                                                   │   │
│  │  • Scheduled (Cron)                                                  │   │
│  │  • Drift Detection Alert                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        PIPELINE STEPS                                │   │
│  │                                                                      │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │   │
│  │  │  Load    │───▶│  Train   │───▶│ Validate │───▶│   Register   │  │   │
│  │  │  Data    │    │  Model   │    │  Model   │    │   (MLflow)   │  │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────┬───────┘  │   │
│  │                                                          │          │   │
│  │                                        ┌─────────────────┘          │   │
│  │                                        ▼                            │   │
│  │                                  ┌──────────┐                       │   │
│  │                                  │  Deploy  │                       │   │
│  │                                  │  Model   │                       │   │
│  │                                  └──────────┘                       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Task 3.1: Kubeflow Setup trên Minikube (1 giờ)

### Install Kubeflow Pipelines

```bash
#!/bin/bash
# scripts/setup-kubeflow.sh

set -e

echo "=========================================="
echo "Installing Kubeflow Pipelines on Minikube"
echo "=========================================="

# Kubeflow Pipelines standalone version
KFP_VERSION=2.0.3

# Install Kubeflow Pipelines
echo "[1/3] Installing Kubeflow Pipelines..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$KFP_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$KFP_VERSION"

# Wait for pods
echo "[2/3] Waiting for Kubeflow Pipelines pods..."
kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s

# Port forward for UI access
echo "[3/3] Setting up port forward..."
echo "Run this command to access Kubeflow UI:"
echo "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"

echo ""
echo "=========================================="
echo "Kubeflow Pipelines Installed!"
echo "=========================================="
echo "UI: http://localhost:8080 (after port-forward)"
echo ""
echo "Install Python SDK:"
echo "  pip install kfp==2.4.0"
```

### Python SDK Requirements

```txt
# kubeflow/requirements.txt
kfp==2.4.0
mlflow==2.9.2
scikit-learn==1.3.2
numpy==1.26.2
```


---

## Task 3.2: Kubeflow Pipeline Definition (2 giờ)

### Pipeline Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPONENTS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │  load_data      │  Output: dataset path                     │
│  │  (lightweight)  │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  train_model    │  Input: dataset                           │
│  │  (sklearn)      │  Output: run_id, metrics                  │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  validate_model │  Input: run_id, metrics                   │
│  │  (threshold)    │  Output: is_valid (bool)                  │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼ (if valid)                                         │
│  ┌─────────────────┐                                           │
│  │  register_model │  Input: run_id                            │
│  │  (MLflow)       │  Output: model_version                    │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  deploy_model   │  Input: model_version                     │
│  │  (K8s rollout)  │  Output: deployment_status                │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Code

```python
# kubeflow/pipeline.py
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Configuration (Minikube)
MLFLOW_TRACKING_URI = "http://mlflow.mlops.svc.cluster.local:5000"
MODEL_NAME = "iris-model"
BASE_IMAGE = "python:3.9-slim"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["scikit-learn==1.3.2", "numpy==1.26.2"]
)
def load_data(dataset: Output[Dataset]):
    """Load training data"""
    from sklearn.datasets import load_iris
    import numpy as np
    import json
    
    iris = load_iris()
    
    # Save data
    np.save(f"{dataset.path}_X.npy", iris.data)
    np.save(f"{dataset.path}_y.npy", iris.target)
    
    # Save metadata
    with open(dataset.path, "w") as f:
        json.dump({
            "n_samples": len(iris.data),
            "n_features": iris.data.shape[1],
            "feature_names": iris.feature_names
        }, f)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "scikit-learn==1.3.2", 
        "numpy==1.26.2", 
        "mlflow==2.9.2"
    ]
)
def train_model(
    dataset: Input[Dataset],
    mlflow_uri: str,
    model_name: str,
    metrics_out: Output[Metrics]
) -> str:
    """Train model and log to MLflow"""
    import mlflow
    import mlflow.sklearn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from datetime import datetime
    
    # Load data
    X = np.load(f"{dataset.path}_X.npy")
    y = np.load(f"{dataset.path}_y.npy")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("kubeflow-retrain")
    
    with mlflow.start_run(run_name=f"kfp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Train
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log to MLflow
        mlflow.log_params({"n_estimators": 100, "random_state": 42})
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
        mlflow.sklearn.log_model(model, "model")
        
        # Save reference data
        np.save("/tmp/reference_data.npy", X_train)
        mlflow.log_artifact("/tmp/reference_data.npy", "drift")
        
        run_id = mlflow.active_run().info.run_id
        
        # Log metrics for Kubeflow
        metrics_out.log_metric("accuracy", accuracy)
        metrics_out.log_metric("f1_score", f1)
        
        return run_id


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["mlflow==2.9.2"])
def validate_model(
    run_id: str,
    mlflow_uri: str,
    model_name: str,
    min_accuracy: float = 0.9
) -> bool:
    """Validate model meets threshold"""
    import mlflow
    
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    # Get run metrics
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    
    print(f"Model accuracy: {accuracy}")
    print(f"Minimum required: {min_accuracy}")
    
    is_valid = accuracy >= min_accuracy
    print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
    
    return is_valid


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["mlflow==2.9.2"])
def register_model(
    run_id: str,
    mlflow_uri: str,
    model_name: str
) -> str:
    """Register model to MLflow and promote to Production"""
    import mlflow
    
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    # Register model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    
    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"Model {model_name} version {result.version} promoted to Production")
    return result.version


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes==28.1.0"]
)
def deploy_model(model_version: str, namespace: str = "mlops") -> str:
    """Trigger Kubernetes deployment rollout"""
    from kubernetes import client, config
    
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
    
    apps_v1 = client.AppsV1Api()
    
    # Trigger rollout restart
    patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": 
                            __import__('datetime').datetime.now().isoformat()
                    }
                }
            }
        }
    }
    
    apps_v1.patch_namespaced_deployment(
        name="ml-api",
        namespace=namespace,
        body=patch
    )
    
    print(f"Deployment ml-api restarted to load model version {model_version}")
    return "success"


@dsl.pipeline(name="ml-retrain-pipeline", description="Retrain and deploy ML model")
def retrain_pipeline(
    mlflow_uri: str = MLFLOW_TRACKING_URI,
    model_name: str = MODEL_NAME,
    min_accuracy: float = 0.9
):
    """Main retraining pipeline"""
    
    # Step 1: Load data
    load_task = load_data()
    
    # Step 2: Train model
    train_task = train_model(
        dataset=load_task.outputs["dataset"],
        mlflow_uri=mlflow_uri,
        model_name=model_name
    )
    
    # Step 3: Validate
    validate_task = validate_model(
        run_id=train_task.output,
        mlflow_uri=mlflow_uri,
        model_name=model_name,
        min_accuracy=min_accuracy
    )
    
    # Step 4: Register (only if valid)
    with dsl.Condition(validate_task.output == True):
        register_task = register_model(
            run_id=train_task.output,
            mlflow_uri=mlflow_uri,
            model_name=model_name
        )
        
        # Step 5: Deploy
        deploy_model(model_version=register_task.output)


# Compile pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=retrain_pipeline,
        package_path="retrain_pipeline.yaml"
    )
    print("Pipeline compiled to retrain_pipeline.yaml")
```


---

## Task 3.3: Drift Detection & Pipeline Trigger (1 giờ)

### Drift Detection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRIFT → RETRAIN FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐                                             │
│   │  ML API      │                                             │
│   │  /predict    │──────┐                                      │
│   └──────────────┘      │                                      │
│                         ▼                                      │
│                  ┌──────────────┐                              │
│                  │ Drift        │                              │
│                  │ Detector     │                              │
│                  │ (KS Test)    │                              │
│                  └──────┬───────┘                              │
│                         │                                      │
│            ┌────────────┴────────────┐                         │
│            ▼                         ▼                         │
│     ┌──────────────┐         ┌──────────────┐                 │
│     │  No Drift    │         │   Drift!     │                 │
│     │  (continue)  │         │  Detected    │                 │
│     └──────────────┘         └──────┬───────┘                 │
│                                     │                          │
│                                     ▼                          │
│                              ┌──────────────┐                  │
│                              │   Trigger    │                  │
│                              │   Kubeflow   │                  │
│                              │   Pipeline   │                  │
│                              └──────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Drift Detector

```python
# api/drift.py
from scipy import stats
import numpy as np
import os

class DriftDetector:
    def __init__(self, reference_data_path: str):
        self.reference = np.load(reference_data_path)
        self.threshold = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
        self.window_size = int(os.getenv("DRIFT_WINDOW_SIZE", "100"))
        self.recent_data = []
    
    def add_sample(self, features: list):
        self.recent_data.append(features)
        if len(self.recent_data) > self.window_size:
            self.recent_data.pop(0)
    
    def check_drift(self) -> dict:
        if len(self.recent_data) < self.window_size:
            return {
                "drift_detected": False,
                "reason": f"Collecting samples ({len(self.recent_data)}/{self.window_size})",
                "samples": len(self.recent_data)
            }
        
        recent = np.array(self.recent_data)
        results = {}
        drift_detected = False
        
        for i in range(recent.shape[1]):
            stat, p_value = stats.ks_2samp(self.reference[:, i], recent[:, i])
            feature_drift = p_value < self.threshold
            results[f"feature_{i}"] = {
                "ks_stat": round(stat, 4),
                "p_value": round(p_value, 4),
                "drift": feature_drift
            }
            if feature_drift:
                drift_detected = True
        
        return {
            "drift_detected": drift_detected,
            "threshold": self.threshold,
            "samples": len(self.recent_data),
            "features": results
        }
    
    def reset(self):
        self.recent_data = []
```

### Pipeline Trigger Script

```python
# kubeflow/trigger.py
import kfp
import argparse
import requests
from datetime import datetime

KFP_HOST = "http://localhost:8080"  # Kubeflow Pipelines UI
PIPELINE_NAME = "ml-retrain-pipeline"


def check_drift(api_url: str) -> bool:
    """Check drift status from ML API"""
    response = requests.get(f"{api_url}/drift/status")
    data = response.json()
    return data.get("drift_detected", False)


def trigger_pipeline(
    kfp_host: str,
    pipeline_name: str,
    mlflow_uri: str,
    model_name: str
):
    """Trigger Kubeflow pipeline"""
    client = kfp.Client(host=kfp_host)
    
    # Find pipeline
    pipelines = client.list_pipelines().pipelines
    pipeline = next((p for p in pipelines if p.display_name == pipeline_name), None)
    
    if not pipeline:
        print(f"Pipeline '{pipeline_name}' not found. Uploading...")
        # Upload pipeline if not exists
        pipeline = client.upload_pipeline(
            pipeline_package_path="retrain_pipeline.yaml",
            pipeline_name=pipeline_name
        )
    
    # Create run
    run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run = client.create_run_from_pipeline_package(
        pipeline_file="retrain_pipeline.yaml",
        run_name=run_name,
        arguments={
            "mlflow_uri": mlflow_uri,
            "model_name": model_name,
            "min_accuracy": 0.9
        }
    )
    
    print(f"Pipeline run created: {run.run_id}")
    print(f"View at: {kfp_host}/#/runs/details/{run.run_id}")
    
    return run.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--kfp-host", default=KFP_HOST)
    parser.add_argument("--mlflow-uri", default="http://mlflow.mlops.svc.cluster.local:5000")
    parser.add_argument("--model-name", default="iris-model")
    parser.add_argument("--force", action="store_true", help="Trigger without drift check")
    args = parser.parse_args()
    
    if args.force:
        print("Force trigger enabled, skipping drift check")
        trigger_pipeline(args.kfp_host, PIPELINE_NAME, args.mlflow_uri, args.model_name)
    else:
        print("Checking drift status...")
        if check_drift(args.api_url):
            print("Drift detected! Triggering retrain pipeline...")
            trigger_pipeline(args.kfp_host, PIPELINE_NAME, args.mlflow_uri, args.model_name)
        else:
            print("No drift detected. Skipping retrain.")


if __name__ == "__main__":
    main()
```

### Scheduled Trigger (CronJob)

```yaml
# k8s/kubeflow/cronjob-trigger.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: drift-check-trigger
  namespace: mlops
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trigger
            image: python:3.9-slim
            command: ["python", "/scripts/trigger.py"]
            args:
              - "--api-url=http://ml-api:8000"
              - "--kfp-host=http://ml-pipeline-ui.kubeflow:80"
              - "--mlflow-uri=http://mlflow.mlops:5000"
            volumeMounts:
            - name: scripts
              mountPath: /scripts
          volumes:
          - name: scripts
            configMap:
              name: trigger-script
          restartPolicy: OnFailure
```


---

## Task 3.4: E2E Test (30 phút)

### Test Script

```bash
#!/bin/bash
# scripts/e2e_test.sh

set -e

API_URL=${1:-"http://localhost:8000"}
KFP_URL=${2:-"http://localhost:8080"}

echo "=========================================="
echo "E2E TEST - MLOps Pipeline"
echo "=========================================="

# 1. Health check
echo "[1/6] Health check..."
curl -sf $API_URL/health | jq .
echo "✓ API healthy"

# 2. Model info
echo ""
echo "[2/6] Model info..."
curl -sf $API_URL/model/info | jq .
echo "✓ Model loaded from MLflow"

# 3. Prediction test
echo ""
echo "[3/6] Testing prediction..."
curl -sf -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' | jq .
echo "✓ Prediction works"

# 4. Simulate drift
echo ""
echo "[4/6] Simulating drift (100 shifted samples)..."
for i in $(seq 1 100); do
  curl -sf -X POST $API_URL/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [7.1, 5.5, 3.4, 2.2]}' > /dev/null
done
echo "✓ Sent 100 shifted samples"

# 5. Check drift
echo ""
echo "[5/6] Checking drift status..."
DRIFT=$(curl -sf $API_URL/drift/status)
echo $DRIFT | jq .

DETECTED=$(echo $DRIFT | jq -r '.drift_detected')
if [ "$DETECTED" = "true" ]; then
  echo "✓ Drift detected!"
  
  # 6. Trigger pipeline
  echo ""
  echo "[6/6] Triggering Kubeflow pipeline..."
  python kubeflow/trigger.py --api-url $API_URL --kfp-host $KFP_URL --force
  echo "✓ Pipeline triggered"
else
  echo "⚠ Drift not detected (may need more samples)"
fi

echo ""
echo "=========================================="
echo "E2E TEST COMPLETE"
echo "=========================================="
```

---

## Deliverables Ngày 3

- [ ] Kubeflow Pipelines installed và accessible
- [ ] Retraining pipeline compiled và uploaded
- [ ] Drift detection hoạt động
- [ ] Pipeline trigger script hoạt động
- [ ] E2E test pass

## Commands Tham Khảo

```bash
# Setup Kubeflow
./scripts/setup-kubeflow.sh

# Compile pipeline
cd kubeflow
python pipeline.py
# Output: retrain_pipeline.yaml

# Upload pipeline to Kubeflow
python -c "
import kfp
client = kfp.Client(host='http://localhost:8080')
client.upload_pipeline('retrain_pipeline.yaml', 'ml-retrain-pipeline')
"

# Trigger pipeline manually
python kubeflow/trigger.py --force

# Check drift and trigger if needed
python kubeflow/trigger.py --api-url http://$(minikube ip):30080

# View Kubeflow UI
open http://localhost:8080

# Run E2E test
./scripts/e2e_test.sh http://$(minikube ip):30080 http://localhost:8080

# View pipeline runs
kubectl get workflows -n kubeflow
```

## Troubleshooting

### Kubeflow không start
```bash
# Check pods
kubectl get pods -n kubeflow

# Check logs
kubectl logs -n kubeflow deployment/ml-pipeline

# Restart
kubectl rollout restart deployment -n kubeflow
```

### Pipeline fails
```bash
# Check workflow status
kubectl get workflows -n kubeflow

# View logs
kubectl logs -n kubeflow <pod-name>

# Check MLflow connection
kubectl exec -it deployment/ml-api -n mlops -- curl http://mlflow:5000/health
```
