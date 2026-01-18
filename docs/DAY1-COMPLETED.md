# DAY 1 - COMPLETED ✅

## Checklist

- [x] Cài đặt Minikube
- [x] Start Minikube cluster
- [x] Tạo namespace `mlops`
- [x] Deploy MLflow server với artifact proxying
- [x] Train model và log to MLflow
- [x] Register model trong MLflow Registry
- [x] Promote model lên Production stage

---

## Các Bước Thực Hiện Chi Tiết

### 1. Cài đặt Minikube

```bash
brew install minikube
```

### 2. Start Minikube Cluster

```bash
# Start với 4GB RAM, 2 CPUs (tối ưu cho host 16GB)
minikube start --memory=4096 --cpus=2 --driver=docker

# Verify
minikube status
kubectl get nodes
```

**Output:**
```
minikube
type: Control Plane
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured

NAME       STATUS   ROLES           AGE   VERSION
minikube   Ready    control-plane   ...   v1.34.0
```

### 3. Tạo Namespace

```bash
kubectl create namespace mlops
```

### 4. Deploy MLflow Server

**Tạo file `k8s/mlflow/pvc.yaml`:**
```yaml
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
```

**Tạo file `k8s/mlflow/deployment.yaml`:**
```yaml
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
          - "--artifacts-destination=/mlflow/artifacts"
          - "--serve-artifacts"  # QUAN TRỌNG: Enable artifact proxying
        ports:
        - containerPort: 5000
        volumeMounts:
        - name: mlflow-data
          mountPath: /mlflow
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
      volumes:
      - name: mlflow-data
        persistentVolumeClaim:
          claimName: mlflow-data
```

**Tạo file `k8s/mlflow/service.yaml`:**
```yaml
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

**Apply:**
```bash
kubectl apply -f k8s/mlflow/
```

**Verify:**
```bash
kubectl get all -n mlops
```


### 5. Access MLflow

```bash
# Port-forward để access từ host
kubectl port-forward svc/mlflow 5000:5000 -n mlops

# Test API
curl http://localhost:5000/api/2.0/mlflow/experiments/search \
  -X POST -H "Content-Type: application/json" \
  -d '{"max_results": 10}'

# Mở UI
open http://localhost:5000
```

### 6. Train Model và Log to MLflow

**Cài dependencies:**
```bash
pip3 install mlflow==2.9.2 scikit-learn numpy
```

**Tạo file `model/train.py`:**
```python
import mlflow
import mlflow.sklearn
import argparse
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "iris-classification"
MODEL_NAME = "iris-model"


def train_model(mlflow_uri: str, register: bool = True):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42,
        }
        mlflow.log_params(params)
        
        # Train
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
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
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", default=MLFLOW_TRACKING_URI)
    parser.add_argument("--register", action="store_true", default=True)
    args = parser.parse_args()
    
    train_model(args.mlflow_uri, args.register)
```

**Run training:**
```bash
python3 model/train.py --mlflow-uri http://localhost:5000
```

**Output:**
```
Loading data...
Training model...
Logging model to MLflow...
Successfully registered model 'iris-model'.
Created version '1' of model 'iris-model'.

==================================================
Training Complete!
==================================================
Run ID: 0ffe549b20524ec8bbe6b730076cacb1
Metrics: {'accuracy': 1.0, 'f1_score': 1.0, 'precision': 1.0, 'recall': 1.0}
Model registered as: iris-model
```

### 7. Promote Model to Production

```bash
python3 -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name='iris-model',
    version='1',
    stage='Production'
)
print('Model iris-model v1 promoted to Production!')
"
```

---

## Kết Quả

| Component | Status | Details |
|-----------|--------|---------|
| Minikube | ✅ Running | 192.168.49.2, 4GB RAM, 2 CPUs |
| MLflow Server | ✅ Running | Pod: mlflow-xxx, Port: 5000 |
| MLflow UI | ✅ Accessible | http://localhost:5000 (port-forward) |
| Model | ✅ Registered | iris-model v1 @ Production |
| Metrics | ✅ Logged | accuracy=1.0, f1=1.0 |

## Files Created

```
├── k8s/
│   └── mlflow/
│       ├── deployment.yaml
│       ├── pvc.yaml
│       └── service.yaml
└── model/
    ├── train.py
    └── requirements.txt
```

## Next Steps (Day 2)

- [ ] Deploy ML API để serve model
- [ ] Setup drift detection endpoint
- [ ] Test prediction API
