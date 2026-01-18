# TỔNG QUAN MLOPS MVP (2-3 NGÀY)

## Kiến Trúc Tổng Quan

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

## Data Flow Chi Tiết

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐      ┌─────────────────────────────────────────────────────┐  │
│  │  Data   │      │                    MLFLOW                           │  │
│  │ Source  │      │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  └────┬────┘      │  │ Experiments │  │   Models    │  │  Artifacts  │  │  │
│       │           │  │  Tracking   │  │  Registry   │  │   Store     │  │  │
│       ▼           │  └─────────────┘  └──────┬──────┘  └─────────────┘  │  │
│  ┌─────────┐      └──────────────────────────┼──────────────────────────┘  │
│  │ Train   │─────────────────────────────────┘                             │
│  │ Script  │                                 │                             │
│  └────┬────┘                                 ▼                             │
│       │                              ┌─────────────┐                       │
│       │                              │   Docker    │                       │
│       │                              │   Build     │                       │
│       │                              └──────┬──────┘                       │
│       │                                     │                              │
│       │           ┌─────────────────────────┼─────────────────────────┐    │
│       │           │                 KUBEFLOW PIPELINES                │    │
│       │           │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │    │
│       └──────────▶│  │  Train   │─▶│ Validate │─▶│ Register Model   │ │    │
│                   │  │Component │  │Component │  │   (MLflow)       │ │    │
│                   │  └──────────┘  └──────────┘  └────────┬─────────┘ │    │
│                   └───────────────────────────────────────┼───────────┘    │
│                                                           │                │
│                                                           ▼                │
│                                                    ┌─────────────┐         │
│                                                    │  K8s Deploy │         │
│                                                    └─────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cấu Trúc Thư Mục

```
mlops-mvp/
├── model/
│   ├── train.py              # Training script với MLflow logging
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── mlflow/
│       ├── deployment.yaml
│       └── service.yaml
├── kubeflow/
│   ├── pipeline.py           # Kubeflow pipeline definition
│   ├── components/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── deploy.py
│   └── trigger.py            # Pipeline trigger script
├── scripts/
│   ├── setup-mlflow.sh
│   ├── setup-kubeflow.sh
│   └── deploy.sh
├── .github/
│   └── workflows/
│       └── ci.yaml
└── README.md
```


## Timeline

| Ngày | Mục tiêu | Deliverables |
|------|----------|--------------|
| Day 1 | MLflow Setup | MLflow server, Model registry, CI validation |
| Day 2 | Deploy & Monitor | K8s deployment (existing image), Metrics |
| Day 3 | Kubeflow & Drift | Kubeflow pipeline, Drift detection, Auto-retrain |

## Lưu Ý
- **Docker image đã có sẵn** - không cần build mới
- Focus vào MLflow integration và Kubeflow pipeline

## Tech Stack

| Component | Tool | Lý do chọn |
|-----------|------|------------|
| Model | Scikit-learn | Simple, fast training |
| Serving | FastAPI | Lightweight, async |
| Container | Docker | Standard |
| Orchestration | **Minikube** | Local K8s, free |
| Model Registry | MLflow | Experiment tracking + Model versioning |
| Pipeline | Kubeflow Pipelines | Native K8s, visual UI |
| CI | GitHub Actions | Free, easy setup |
| Drift Detection | Scipy (KS test) | Simple, no extra deps |

## Prerequisites

```bash
# Required tools
- Docker
- kubectl
- minikube
- Python 3.9+
- pip
```

## Quick Architecture Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                    MINIKUBE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Developer                                                     │
│      │                                                          │
│      ▼                                                          │
│   ┌──────────┐     ┌──────────┐     ┌──────────────────┐       │
│   │  GitHub  │────▶│    CI    │────▶│  Docker Registry │       │
│   │   Push   │     │ Actions  │     │    (local)       │       │
│   └──────────┘     └──────────┘     └────────┬─────────┘       │
│                                               │                 │
│   ┌───────────────────────────────────────────┼────────────┐   │
│   │                 MINIKUBE CLUSTER          │            │   │
│   │                                           ▼            │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │   │
│   │  │  MLflow  │  │ Kubeflow │  │   ML Model API   │     │   │
│   │  │  Server  │  │Pipelines │  │   (FastAPI)      │     │   │
│   │  │  :5000   │  │  :8080   │  │     :8000        │     │   │
│   │  └────┬─────┘  └──────────┘  └──────────────────┘     │   │
│   │       │                                               │   │
│   │       ▼                                               │   │
│   │  ┌──────────────┐                                     │   │
│   │  │     PVC      │                                     │   │
│   │  │  (Artifacts) │                                     │   │
│   │  └──────────────┘                                     │   │
│   │                                                       │   │
│   └───────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
