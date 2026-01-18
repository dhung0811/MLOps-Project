# BÁO CÁO HỌC THUẬT
# XÂY DỰNG HỆ THỐNG MLOPS MVP CHO TRIỂN KHAI VÀ GIÁM SÁT MÔ HÌNH HỌC MÁY

---

## TÓM TẮT

Báo cáo này trình bày việc thiết kế và triển khai một hệ thống MLOps (Machine Learning Operations) MVP (Minimum Viable Product) hoàn chỉnh, bao gồm các thành phần: quản lý thí nghiệm với MLflow, triển khai mô hình trên Kubernetes (Minikube), phát hiện data drift, và tự động hóa quy trình huấn luyện lại với Kubeflow Pipelines. Hệ thống được thiết kế để chạy trên môi trường local với tài nguyên hạn chế (16GB RAM), phù hợp cho mục đích học tập và phát triển prototype.

**Từ khóa:** MLOps, MLflow, Kubernetes, Kubeflow Pipelines, Data Drift Detection, Machine Learning Deployment

---

## MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Phương pháp triển khai](#4-phương-pháp-triển-khai)
5. [Kết quả thực nghiệm](#5-kết-quả-thực-nghiệm)
6. [Thảo luận](#6-thảo-luận)
7. [Kết luận và hướng phát triển](#7-kết-luận-và-hướng-phát-triển)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. GIỚI THIỆU

### 1.1. Đặt vấn đề

Trong bối cảnh ứng dụng Machine Learning (ML) ngày càng phổ biến trong các hệ thống production, việc quản lý vòng đời của mô hình ML trở thành một thách thức lớn. Các vấn đề thường gặp bao gồm:

- **Thiếu khả năng tái tạo (Reproducibility):** Khó khăn trong việc tái tạo kết quả thí nghiệm do không theo dõi đầy đủ các tham số và phiên bản dữ liệu.
- **Triển khai thủ công:** Quy trình triển khai mô hình phức tạp, dễ xảy ra lỗi và tốn thời gian.
- **Data Drift:** Hiệu suất mô hình suy giảm theo thời gian do sự thay đổi trong phân phối dữ liệu đầu vào.
- **Thiếu tự động hóa:** Quy trình huấn luyện lại mô hình thường được thực hiện thủ công, không kịp thời phản ứng với sự thay đổi của dữ liệu.

### 1.2. Mục tiêu nghiên cứu

Nghiên cứu này nhằm xây dựng một hệ thống MLOps MVP với các mục tiêu cụ thể:

1. Thiết lập hệ thống theo dõi thí nghiệm và quản lý phiên bản mô hình với MLflow
2. Triển khai mô hình ML trên Kubernetes với khả năng mở rộng
3. Xây dựng cơ chế phát hiện data drift sử dụng Kolmogorov-Smirnov test
4. Tự động hóa quy trình huấn luyện lại với Kubeflow Pipelines

### 1.3. Phạm vi nghiên cứu

- **Môi trường:** Local development với Minikube
- **Mô hình mẫu:** Random Forest Classifier cho bài toán phân loại Iris
- **Thời gian triển khai:** 3 ngày (MVP approach)

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. MLOps - Machine Learning Operations

MLOps là một tập hợp các thực hành kết hợp Machine Learning, DevOps và Data Engineering nhằm triển khai và duy trì các hệ thống ML trong production một cách đáng tin cậy và hiệu quả (Kreuzberger et al., 2023).

Các nguyên tắc cốt lõi của MLOps bao gồm:
- **Automation:** Tự động hóa các quy trình từ huấn luyện đến triển khai
- **Continuous Integration/Continuous Deployment (CI/CD):** Tích hợp và triển khai liên tục
- **Monitoring:** Giám sát hiệu suất mô hình trong production
- **Versioning:** Quản lý phiên bản cho code, data và model

### 2.2. MLflow - Nền tảng quản lý vòng đời ML

MLflow là một nền tảng mã nguồn mở để quản lý vòng đời ML end-to-end, bao gồm bốn thành phần chính:

1. **MLflow Tracking:** Ghi lại các tham số, metrics và artifacts của mỗi lần chạy thí nghiệm
2. **MLflow Projects:** Đóng gói code ML theo định dạng có thể tái sử dụng
3. **MLflow Models:** Quản lý và triển khai mô hình với nhiều framework khác nhau
4. **MLflow Model Registry:** Quản lý phiên bản mô hình với các stage (Staging, Production)

### 2.3. Kubernetes và Container Orchestration

Kubernetes là một hệ thống mã nguồn mở để tự động hóa việc triển khai, mở rộng và quản lý các ứng dụng container. Trong ngữ cảnh MLOps, Kubernetes cung cấp:

- **Scalability:** Khả năng mở rộng theo nhu cầu
- **High Availability:** Đảm bảo tính sẵn sàng cao
- **Resource Management:** Quản lý tài nguyên hiệu quả
- **Service Discovery:** Tự động phát hiện và kết nối các service

### 2.4. Data Drift Detection

Data drift xảy ra khi phân phối của dữ liệu đầu vào trong production khác biệt so với dữ liệu huấn luyện. Nghiên cứu này sử dụng **Kolmogorov-Smirnov (KS) test** để phát hiện drift:

$$D_n = \sup_x |F_n(x) - F(x)|$$

Trong đó:
- $F_n(x)$: Hàm phân phối tích lũy thực nghiệm của mẫu
- $F(x)$: Hàm phân phối tích lũy tham chiếu
- $D_n$: Thống kê KS

Nếu p-value < threshold (mặc định 0.05), drift được phát hiện.

### 2.5. Kubeflow Pipelines

Kubeflow Pipelines là một nền tảng để xây dựng và triển khai các ML workflow có thể tái sử dụng, dựa trên Docker containers. Các đặc điểm chính:

- **Declarative Pipelines:** Định nghĩa pipeline bằng Python DSL
- **Reusable Components:** Các component có thể tái sử dụng
- **Experiment Tracking:** Theo dõi các lần chạy pipeline
- **Visualization:** Giao diện trực quan để theo dõi tiến trình

---

## 3. KIẾN TRÚC HỆ THỐNG

### 3.1. Tổng quan kiến trúc

Hệ thống được thiết kế theo kiến trúc microservices, triển khai trên Minikube cluster với các thành phần chính:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOPS PIPELINE MVP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
│   │  TRAIN   │───▶│  IMAGE   │───▶│ REGISTRY │───▶│  CI VALIDATION   │     │
│   │ (Model)  │    │ (Docker) │    │ (MLflow) │    │ (GitHub Actions) │     │
│   └──────────┘    └──────────┘    └──────────┘    └────────┬─────────┘     │
│                                                             │               │
│                                                             ▼               │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
│   │ RETRAIN  │◀───│  DRIFT   │◀───│ MONITOR  │◀───│     DEPLOY       │     │
│   │(Kubeflow)│    │(KS Test) │    │(Metrics) │    │   (Kubernetes)   │     │
│   └────┬─────┘    └──────────┘    └──────────┘    └──────────────────┘     │
│        │                                                                    │
│        └────────────────────▶ Quay lại TRAIN ──────────────────────────────┘
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Các thành phần hệ thống

#### 3.2.1. MLflow Server
- **Chức năng:** Tracking experiments, Model Registry, Artifact Store
- **Deployment:** Kubernetes Deployment với PersistentVolumeClaim
- **Endpoint:** ClusterIP (internal) + NodePort (external access)

#### 3.2.2. ML Model API
- **Framework:** FastAPI
- **Chức năng:** Serving predictions, Drift detection, Metrics exposure
- **Endpoints:**
  - `/predict`: Inference endpoint
  - `/drift/status`: Drift detection status
  - `/metrics`: Prometheus-compatible metrics
  - `/health`: Health check

#### 3.2.3. Kubeflow Pipelines
- **Chức năng:** Orchestrate retraining workflow
- **Components:** Load Data → Train → Validate → Register → Deploy

### 3.3. Luồng dữ liệu (Data Flow)

```
┌─────────┐      ┌─────────────────────────────────────────────────────┐
│  Data   │      │                    MLFLOW                           │
│ Source  │      │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
└────┬────┘      │  │ Experiments │  │   Models    │  │  Artifacts  │  │
     │           │  │  Tracking   │  │  Registry   │  │   Store     │  │
     ▼           │  └─────────────┘  └──────┬──────┘  └─────────────┘  │
┌─────────┐      └──────────────────────────┼──────────────────────────┘
│ Train   │─────────────────────────────────┘
│ Script  │                                 │
└────┬────┘                                 ▼
     │                              ┌─────────────┐
     │                              │   Docker    │
     │                              │   Build     │
     │                              └──────┬──────┘
     │                                     │
     │           ┌─────────────────────────┼─────────────────────────┐
     │           │                 KUBEFLOW PIPELINES                │
     │           │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
     └──────────▶│  │  Train   │─▶│ Validate │─▶│ Register Model   │ │
                 │  │Component │  │Component │  │   (MLflow)       │ │
                 │  └──────────┘  └──────────┘  └────────┬─────────┘ │
                 └───────────────────────────────────────┼───────────┘
                                                         │
                                                         ▼
                                                  ┌─────────────┐
                                                  │  K8s Deploy │
                                                  └─────────────┘
```

### 3.4. Network Architecture

| Service | Type | Internal Port | External Port | Access |
|---------|------|---------------|---------------|--------|
| mlflow | ClusterIP | 5000 | - | Internal |
| mlflow-external | NodePort | 5000 | 30500 | External |
| ml-api | NodePort | 8000 | 30080 | External |
| ml-pipeline-ui | ClusterIP | 80 | 8080 (port-forward) | External |

---

## 4. PHƯƠNG PHÁP TRIỂN KHAI

### 4.1. Giai đoạn 1: MLflow Setup & Integration (Ngày 1)

#### 4.1.1. Triển khai MLflow Server

MLflow được triển khai trên Kubernetes với cấu hình sau:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlops
spec:
  replicas: 1
  template:
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
          - "--serve-artifacts"
```

#### 4.1.2. Training Script với MLflow Integration

Script huấn luyện được thiết kế để tự động log các thông tin quan trọng:

```python
def train_model(register: bool = True):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        })
        
        # Log model to registry
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        
        # Log reference data for drift detection
        mlflow.log_artifact("/tmp/reference_data.npy", artifact_path="drift")
```

### 4.2. Giai đoạn 2: Deployment & Monitoring (Ngày 2)

#### 4.2.1. ML API với FastAPI

API được thiết kế với các endpoint cần thiết cho production:

```python
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    features = np.array([request.features])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].tolist()
    
    # Track for drift detection
    drift_detector.add_sample(request.features)
    
    return PredictResponse(
        prediction=int(prediction),
        probability=probability,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
        latency_ms=latency
    )
```

#### 4.2.2. Metrics Endpoint

Metrics được expose theo định dạng Prometheus:

```
# HELP prediction_requests_total Total predictions
# TYPE prediction_requests_total counter
prediction_requests_total 1234

# HELP prediction_latency_avg_ms Average latency
# TYPE prediction_latency_avg_ms gauge
prediction_latency_avg_ms 5.23
```

### 4.3. Giai đoạn 3: Kubeflow & Drift Detection (Ngày 3)

#### 4.3.1. Drift Detection Implementation

```python
class DriftDetector:
    def __init__(self, reference_data_path: str):
        self.reference = np.load(reference_data_path)
        self.threshold = 0.05
        self.window_size = 100
        self.recent_data = []
    
    def check_drift(self) -> dict:
        if len(self.recent_data) < self.window_size:
            return {"drift_detected": False, "reason": "Collecting samples"}
        
        recent = np.array(self.recent_data)
        drift_detected = False
        
        for i in range(recent.shape[1]):
            stat, p_value = stats.ks_2samp(
                self.reference[:, i], 
                recent[:, i]
            )
            if p_value < self.threshold:
                drift_detected = True
        
        return {"drift_detected": drift_detected}
```

#### 4.3.2. Kubeflow Pipeline Definition

```python
@dsl.pipeline(name="ml-retrain-pipeline")
def retrain_pipeline(mlflow_uri: str, model_name: str, min_accuracy: float = 0.9):
    # Step 1: Load data
    load_task = load_data()
    
    # Step 2: Train model
    train_task = train_model(dataset=load_task.outputs["dataset"])
    
    # Step 3: Validate
    validate_task = validate_model(run_id=train_task.output)
    
    # Step 4: Register (conditional)
    with dsl.Condition(validate_task.output == True):
        register_task = register_model(run_id=train_task.output)
        deploy_model(model_version=register_task.output)
```

---

## 5. KẾT QUẢ THỰC NGHIỆM

### 5.1. Kết quả triển khai

| Component | Status | Chi tiết |
|-----------|--------|----------|
| Minikube Cluster | ✅ Running | 4GB RAM, 2 CPUs |
| MLflow Server | ✅ Running | Port 30500 |
| ML API | ✅ Running | Port 30080 |
| Model Registry | ✅ Active | iris-model v1 @ Production |
| Kubeflow Pipelines | ✅ Installed | Port 8080 |

### 5.2. Hiệu suất mô hình

Mô hình Random Forest Classifier đạt được các metrics sau trên tập test:

| Metric | Giá trị |
|--------|---------|
| Accuracy | 1.0 |
| F1-Score (weighted) | 1.0 |
| Precision (weighted) | 1.0 |
| Recall (weighted) | 1.0 |

*Lưu ý: Kết quả cao do sử dụng dataset Iris đơn giản cho mục đích demo.*

### 5.3. Resource Utilization

| Component | Memory Request | Memory Limit | CPU Request |
|-----------|----------------|--------------|-------------|
| MLflow | 256Mi | 1Gi | 250m |
| ML API | 256Mi | 512Mi | 250m |
| Kubeflow | ~2GB | - | 1 core |
| **Total** | ~4-5GB | - | ~2 cores |

### 5.4. API Performance

- **Average Latency:** < 10ms per prediction
- **Throughput:** Có thể xử lý hàng trăm requests/second (single pod)

---

## 6. THẢO LUẬN

### 6.1. Ưu điểm của hệ thống

1. **Reproducibility:** MLflow tracking đảm bảo mọi thí nghiệm có thể tái tạo với đầy đủ parameters, metrics và artifacts.

2. **Automation:** Quy trình từ training đến deployment được tự động hóa thông qua CI/CD và Kubeflow Pipelines.

3. **Monitoring:** Hệ thống có khả năng phát hiện data drift và tự động trigger retraining.

4. **Scalability:** Kiến trúc Kubernetes cho phép mở rộng theo nhu cầu.

5. **Cost-effective:** Sử dụng Minikube cho phép phát triển và test trên môi trường local mà không cần cloud resources.

### 6.2. Hạn chế

1. **Single Node:** Minikube chỉ chạy trên single node, không phản ánh đầy đủ môi trường production multi-node.

2. **Storage:** Sử dụng SQLite và local PVC, không phù hợp cho production với yêu cầu high availability.

3. **Security:** Chưa implement đầy đủ các biện pháp bảo mật như RBAC, mTLS, Secrets Management.

4. **Monitoring Stack:** Chưa tích hợp Prometheus + Grafana cho monitoring toàn diện.

### 6.3. So sánh với các giải pháp khác

| Tiêu chí | Giải pháp này | Cloud-based MLOps |
|----------|---------------|-------------------|
| Chi phí | Miễn phí (local) | Tốn phí cloud |
| Scalability | Hạn chế | Cao |
| Setup complexity | Trung bình | Cao |
| Production-ready | Không | Có |
| Learning curve | Thấp | Cao |

---

## 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 7.1. Kết luận

Nghiên cứu đã thành công trong việc xây dựng một hệ thống MLOps MVP hoàn chỉnh với các thành phần:

- **MLflow** cho experiment tracking và model registry
- **Kubernetes (Minikube)** cho container orchestration
- **FastAPI** cho model serving với drift detection
- **Kubeflow Pipelines** cho automated retraining

Hệ thống đáp ứng được các yêu cầu cơ bản của một MLOps pipeline và phù hợp cho mục đích học tập, prototyping và phát triển trước khi triển khai lên môi trường production.

### 7.2. Hướng phát triển

1. **Production MLflow:** Chuyển sang MySQL/PostgreSQL backend và S3/MinIO artifact store

2. **Monitoring Stack:** Tích hợp Prometheus + Grafana cho observability toàn diện

3. **Advanced Drift Detection:** Sử dụng Evidently AI hoặc WhyLabs cho drift detection nâng cao

4. **GitOps:** Triển khai ArgoCD cho GitOps-based deployments

5. **Auto-scaling:** Implement Horizontal Pod Autoscaler (HPA) dựa trên request volume

6. **Security Hardening:** Implement RBAC, Network Policies, và Secrets Management

7. **Multi-model Serving:** Mở rộng để hỗ trợ nhiều mô hình với A/B testing

---

## 8. TÀI LIỆU THAM KHẢO

1. Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine Learning Operations (MLOps): Overview, Definition, and Architecture. *IEEE Access*, 11, 31866-31879.

2. MLflow Documentation. (2024). MLflow: A Machine Learning Lifecycle Platform. https://mlflow.org/docs/latest/index.html

3. Kubernetes Documentation. (2024). Kubernetes: Production-Grade Container Orchestration. https://kubernetes.io/docs/

4. Kubeflow Documentation. (2024). Kubeflow Pipelines. https://www.kubeflow.org/docs/components/pipelines/

5. Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *Advances in Neural Information Processing Systems*, 28.

6. Amershi, S., et al. (2019). Software Engineering for Machine Learning: A Case Study. *IEEE/ACM 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP)*.

7. FastAPI Documentation. (2024). FastAPI: Modern, Fast Web Framework for Building APIs. https://fastapi.tiangolo.com/

8. Scipy Documentation. (2024). scipy.stats.ks_2samp. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

---

## PHỤ LỤC

### A. Cấu trúc thư mục dự án

```
mlops-mvp/
├── model/
│   ├── train.py              # Training script với MLflow logging
│   └── requirements.txt
├── k8s/
│   ├── mlflow/
│   │   ├── deployment.yaml
│   │   ├── pvc.yaml
│   │   └── service.yaml
│   └── api/
│       ├── deployment.yaml
│       ├── service.yaml
│       └── configmap.yaml
├── kubeflow/
│   ├── pipeline.py           # Kubeflow pipeline definition
│   └── trigger.py            # Pipeline trigger script
├── scripts/
│   ├── setup-cluster.sh
│   ├── setup-kubeflow.sh
│   └── deploy-api.sh
├── .github/
│   └── workflows/
│       └── ci.yaml
└── docs/
    ├── 00-OVERVIEW.md
    ├── 01-DAY1-CORE-PIPELINE.md
    ├── 02-DAY2-DEPLOY-MONITORING.md
    ├── 03-DAY3-KUBEFLOW-DRIFT.md
    └── 04-SUMMARY.md
```

### B. Tech Stack Summary

| Component | Tool | Version | Lý do chọn |
|-----------|------|---------|------------|
| Model | Scikit-learn | 1.3.2 | Simple, fast training |
| Serving | FastAPI | - | Lightweight, async |
| Container | Docker | - | Standard |
| Orchestration | Minikube | - | Local K8s, free |
| Model Registry | MLflow | 2.9.2 | Experiment tracking + Model versioning |
| Pipeline | Kubeflow Pipelines | 2.0.3 | Native K8s, visual UI |
| CI | GitHub Actions | - | Free, easy setup |
| Drift Detection | Scipy (KS test) | - | Simple, no extra deps |

### C. Commands Reference

```bash
# Setup
minikube start --memory=4096 --cpus=2
kubectl apply -f k8s/mlflow/

# Training
python model/train.py --mlflow-uri http://localhost:5000

# Access URLs
echo "MLflow: http://$(minikube ip):30500"
echo "API: http://$(minikube ip):30080"

# Test API
curl -X POST http://$(minikube ip):30080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Trigger retraining
python kubeflow/trigger.py --force
```

---

*Báo cáo được hoàn thành vào tháng 01/2026*
