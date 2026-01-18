# NGÀY 2: DEPLOY & MONITORING (Minikube)

## Mục Tiêu
- Deploy MLflow server lên Minikube
- Deploy ML Model API (sử dụng Docker image có sẵn)
- Setup monitoring cơ bản

> **Note**: Sử dụng Docker image đã có sẵn, chỉ cần config environment variables

## Task 2.1: MLflow trên Minikube (1.5 giờ)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MINIKUBE CLUSTER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    NAMESPACE: mlops                      │   │
│  │                                                          │   │
│  │  ┌──────────────────┐      ┌──────────────────────────┐ │   │
│  │  │  MLFLOW SERVER   │      │     ML MODEL API         │ │   │
│  │  │                  │      │                          │ │   │
│  │  │  ┌────────────┐  │      │  ┌────────┐ ┌────────┐  │ │   │
│  │  │  │ Deployment │  │      │  │ Pod 1  │ │ Pod 2  │  │ │   │
│  │  │  │  (1 pod)   │  │◀─────│  │        │ │        │  │ │   │
│  │  │  └────────────┘  │      │  └────────┘ └────────┘  │ │   │
│  │  │        │         │      │         │               │ │   │
│  │  │  ┌────────────┐  │      │  ┌──────────────────┐   │ │   │
│  │  │  │  Service   │  │      │  │     Service      │   │ │   │
│  │  │  │NodePort    │  │      │  │  NodePort:30080  │   │ │   │
│  │  │  │  :30500    │  │      │  └──────────────────┘   │ │   │
│  │  │  └────────────┘  │      └──────────────────────────┘ │   │
│  │  └──────────────────┘                                   │   │
│  │           │                                              │   │
│  │           ▼                                              │   │
│  │  ┌──────────────────┐                                   │   │
│  │  │       PVC        │                                   │   │
│  │  │  (mlflow-data)   │                                   │   │
│  │  │  - artifacts     │                                   │   │
│  │  │  - mlruns        │                                   │   │
│  │  └──────────────────┘                                   │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MLflow Minikube Manifests

```yaml
# k8s/mlflow/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlops
---
# k8s/mlflow/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-data
  namespace: mlops
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
# k8s/mlflow/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.9.2
        command: ["mlflow", "server"]
        args:
          - "--host=0.0.0.0"
          - "--port=5000"
          - "--backend-store-uri=sqlite:///mlflow/mlflow.db"
          - "--default-artifact-root=/mlflow/artifacts"
        ports:
        - containerPort: 5000
        volumeMounts:
        - name: mlflow-data
          mountPath: /mlflow
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: mlflow-data
        persistentVolumeClaim:
          claimName: mlflow-data
---
# k8s/mlflow/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: mlops
spec:
  type: ClusterIP
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
---
# k8s/mlflow/service-external.yaml (NodePort for minikube)
apiVersion: v1
kind: Service
metadata:
  name: mlflow-external
  namespace: mlops
spec:
  type: NodePort
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
    nodePort: 30500
```


---

## Task 2.2: ML Model API Deployment (1.5 giờ)

### FastAPI Application

```python
# api/main.py
import os
import time
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from model_loader import load_model_from_registry, get_model_info, load_reference_data
from drift import DriftDetector

app = FastAPI(title="ML Model API", version="1.0.0")

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "iris-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Load model at startup
model = None
drift_detector = None
startup_time = time.time()

# Metrics
metrics = {
    "prediction_requests_total": 0,
    "prediction_latency_sum": 0.0,
    "prediction_errors_total": 0
}


@app.on_event("startup")
async def startup_event():
    global model, drift_detector
    print("Loading model from MLflow...")
    model = load_model_from_registry()
    
    print("Loading reference data for drift detection...")
    ref_data_path = load_reference_data()
    drift_detector = DriftDetector(ref_data_path)
    
    print("Startup complete!")


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: int
    probability: List[float]
    model_name: str
    model_stage: str
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "uptime_seconds": round(time.time() - startup_time, 2)
    }


@app.get("/model/info")
def model_info():
    info = get_model_info()
    info["uptime_seconds"] = round(time.time() - startup_time, 2)
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    
    try:
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Expected 4 features")
        
        features = np.array([request.features])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        # Track for drift
        drift_detector.add_sample(request.features)
        
        latency = (time.time() - start) * 1000
        metrics["prediction_requests_total"] += 1
        metrics["prediction_latency_sum"] += latency
        
        return PredictResponse(
            prediction=int(prediction),
            probability=probability,
            model_name=MODEL_NAME,
            model_stage=MODEL_STAGE,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        metrics["prediction_errors_total"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/status")
def drift_status():
    return drift_detector.check_drift()


@app.get("/metrics")
def get_metrics():
    avg_latency = (
        metrics["prediction_latency_sum"] / metrics["prediction_requests_total"]
        if metrics["prediction_requests_total"] > 0 else 0
    )
    
    return f"""# HELP prediction_requests_total Total predictions
# TYPE prediction_requests_total counter
prediction_requests_total {metrics["prediction_requests_total"]}

# HELP prediction_errors_total Total errors
# TYPE prediction_errors_total counter
prediction_errors_total {metrics["prediction_errors_total"]}

# HELP prediction_latency_avg_ms Average latency
# TYPE prediction_latency_avg_ms gauge
prediction_latency_avg_ms {avg_latency:.2f}

# HELP model_info Model information
# TYPE model_info gauge
model_info{{name="{MODEL_NAME}",stage="{MODEL_STAGE}"}} 1
"""
```

### Kubernetes Manifests cho API (Minikube)

> **Note**: Thay `YOUR_IMAGE:TAG` bằng Docker image của bạn

```yaml
# k8s/api/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
  namespace: mlops
data:
  MLFLOW_TRACKING_URI: "http://mlflow:5000"
  MODEL_NAME: "iris-model"
  MODEL_STAGE: "Production"
  LOG_LEVEL: "INFO"
  DRIFT_THRESHOLD: "0.05"
  DRIFT_WINDOW_SIZE: "100"
---
# k8s/api/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: mlops
spec:
  replicas: 1  # Giảm từ 2 xuống 1 cho máy 16GB
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: YOUR_IMAGE:TAG  # <-- Thay bằng image của bạn
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ml-api-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
---
# k8s/api/service.yaml (NodePort for minikube)
apiVersion: v1
kind: Service
metadata:
  name: ml-api
  namespace: mlops
spec:
  type: NodePort
  selector:
    app: ml-api
  ports:
  - port: 8000
    targetPort: 8000
    nodePort: 30080
```


---

## Task 2.3: Deploy Scripts (1 giờ)

### Setup Minikube Script

```bash
#!/bin/bash
# scripts/setup-cluster.sh

set -e

echo "=========================================="
echo "Setting up MLOps Cluster (Minikube)"
echo "=========================================="

# Start minikube if not running
if ! minikube status &> /dev/null; then
    echo "Starting minikube..."
    minikube start --memory=4096 --cpus=2  # Tối ưu cho 16GB host
fi

# Create namespace
echo "[1/4] Creating namespace..."
kubectl apply -f k8s/mlflow/namespace.yaml

# Deploy MLflow
echo "[2/4] Deploying MLflow..."
kubectl apply -f k8s/mlflow/pvc.yaml
kubectl apply -f k8s/mlflow/deployment.yaml
kubectl apply -f k8s/mlflow/service.yaml
kubectl apply -f k8s/mlflow/service-external.yaml

# Wait for MLflow
echo "[3/4] Waiting for MLflow to be ready..."
kubectl wait --for=condition=ready pod -l app=mlflow -n mlops --timeout=180s

# Get MLflow URL
MLFLOW_URL="http://$(minikube ip):30500"

echo "[4/4] Training initial model..."
export MLFLOW_TRACKING_URI=$MLFLOW_URL
python model/train.py --mlflow-uri $MLFLOW_URL

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "MLflow UI: $MLFLOW_URL"
```

### Deploy API Script

```bash
#!/bin/bash
# scripts/deploy-api.sh

set -e

IMAGE=${1:-"YOUR_IMAGE:TAG"}

echo "=========================================="
echo "Deploying ML API to Minikube"
echo "Image: $IMAGE"
echo "=========================================="

# Load image to minikube (if local image)
echo "[1/4] Loading image to minikube..."
minikube image load $IMAGE 2>/dev/null || echo "Image may already exist or is remote"

# Deploy
echo "[2/4] Deploying ML API..."
kubectl apply -f k8s/api/configmap.yaml
sed "s|YOUR_IMAGE:TAG|${IMAGE}|g" k8s/api/deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/api/service.yaml

# Wait for rollout
echo "[3/4] Waiting for rollout..."
kubectl rollout status deployment/ml-api -n mlops --timeout=180s

# Get URL
API_URL="http://$(minikube ip):30080"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "API URL: $API_URL"
echo ""
echo "Test commands:"
echo "  curl $API_URL/health"
echo "  curl $API_URL/model/info"
```

---

## Deliverables Ngày 2

- [ ] Minikube cluster running
- [ ] MLflow server deployed với NodePort
- [ ] ML API deployed với existing image
- [ ] Endpoints hoạt động: `/health`, `/predict`, `/model/info`, `/metrics`

## Commands Tham Khảo

```bash
# Setup minikube cluster
./scripts/setup-cluster.sh

# Deploy API với image của bạn
./scripts/deploy-api.sh your-image:tag

# Check status
kubectl get all -n mlops

# Get URLs
echo "MLflow: http://$(minikube ip):30500"
echo "API: http://$(minikube ip):30080"

# View logs
kubectl logs -f deployment/mlflow -n mlops
kubectl logs -f deployment/ml-api -n mlops

# Test API
API_URL="http://$(minikube ip):30080"
curl $API_URL/health
curl $API_URL/model/info
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Port forward (alternative)
kubectl port-forward svc/mlflow 5000:5000 -n mlops &
kubectl port-forward svc/ml-api 8000:8000 -n mlops &
```
