# TỔNG KẾT & CHECKLIST

## Final Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MINIKUBE MLOPS ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        MINIKUBE CLUSTER                              │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │   │
│  │  │   MLFLOW     │  │   KUBEFLOW   │  │        ML API              │ │   │
│  │  │   SERVER     │  │  PIPELINES   │  │       (FastAPI)            │ │   │
│  │  │              │  │              │  │                            │ │   │
│  │  │ • Tracking   │  │ • Pipeline   │  │ • /predict                 │ │   │
│  │  │ • Registry   │  │   Runs       │  │ • /drift/status            │ │   │
│  │  │   :30500     │  │   :8080      │  │   :30080                   │ │   │
│  │  └──────┬───────┘  └──────┬───────┘  └─────────────┬──────────────┘ │   │
│  │         │                 │                        │                │   │
│  │         │    ┌────────────┴────────────┐          │                │   │
│  │         │    │                         │          │                │   │
│  │         │    ▼                         ▼          │                │   │
│  │         │  ┌─────────────────────────────────┐    │                │   │
│  │         │  │      RETRAINING PIPELINE        │    │                │   │
│  │         │  │  Load → Train → Validate →      │    │                │   │
│  │         │  │  Register → Deploy              │    │                │   │
│  │         │  └─────────────────────────────────┘    │                │   │
│  │         │                 │                       │                │   │
│  │         └─────────────────┴───────────────────────┘                │   │
│  │                           │                                        │   │
│  │                    ┌──────▼──────┐                                 │   │
│  │                    │     PVC     │                                 │   │
│  │                    │  (Storage)  │                                 │   │
│  │                    └─────────────┘                                 │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  EXTERNAL:                                                                  │
│  ┌──────────────┐  ┌──────────────┐                                        │
│  │   GitHub     │  │    Local     │                                        │
│  │   Actions    │  │   Docker     │                                        │
│  │   (CI/CD)    │  │   Images     │                                        │
│  └──────────────┘  └──────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Checklist Hoàn Thành

### Day 1: MLflow Setup
- [ ] MLflow server chạy local
- [ ] Training script với MLflow logging
- [ ] Model registered trong MLflow Registry
- [ ] CI pipeline pass

### Day 2: Deploy & Monitoring
- [ ] MLflow deployed trên Kubernetes
- [ ] ML API deployed (existing image)
- [ ] Model loaded từ MLflow Registry
- [ ] Endpoints hoạt động

### Day 3: Kubeflow & Drift
- [ ] Kubeflow Pipelines installed
- [ ] Retraining pipeline compiled
- [ ] Pipeline uploaded và runnable
- [ ] Drift detection hoạt động
- [ ] E2E test pass

## Key Metrics to Monitor

| Metric | Type | Purpose |
|--------|------|---------|
| `prediction_requests_total` | Counter | Traffic |
| `prediction_latency_avg_ms` | Gauge | Performance |
| `prediction_errors_total` | Counter | Errors |
| `drift_detected` | Boolean | Data drift |
| `model_version` | Info | Tracking |

## Quick Reference Commands

```bash
# === SETUP ===
minikube start --memory=4096 --cpus=2
./scripts/setup-cluster.sh
./scripts/setup-kubeflow.sh

# === DEPLOY ===
./scripts/deploy-api.sh your-image:tag

# === GET URLs ===
echo "MLflow: http://$(minikube ip):30500"
echo "API: http://$(minikube ip):30080"
echo "Kubeflow: http://localhost:8080 (after port-forward)"

# === TEST ===
API_URL="http://$(minikube ip):30080"
curl $API_URL/health
curl $API_URL/model/info
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# === MONITORING ===
kubectl get all -n mlops
kubectl logs -f deployment/ml-api -n mlops

# === KUBEFLOW UI ===
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# === RETRAIN ===
python kubeflow/trigger.py --force
```


## File Structure Created

```
docs/
├── 00-OVERVIEW.md           # Tổng quan, architecture, folder structure
├── 01-DAY1-CORE-PIPELINE.md # MLflow training, Docker, CI
├── 02-DAY2-DEPLOY-MONITORING.md # K8s deployment, API, monitoring
├── 03-DAY3-KUBEFLOW-DRIFT.md    # Kubeflow pipeline, drift detection
└── 04-SUMMARY.md            # Checklist, commands reference
```

## What's INCLUDED vs SKIPPED

| Component | INCLUDED (MVP) | SKIPPED |
|-----------|----------------|---------|
| Docker Image | **Có sẵn** | Build mới |
| Infrastructure | **Minikube** (local, free) | Cloud (GKE/EKS) |
| Model Registry | MLflow (local PVC) | MLflow với Cloud SQL |
| Experiment Tracking | Basic metrics/params | Model signatures, input examples |
| CI/CD | GitHub Actions | ArgoCD, GitOps |
| Deployment | kubectl, basic manifests | Helm charts |
| Monitoring | /metrics endpoint | Prometheus + Grafana stack |
| Drift Detection | KS Test (scipy) | Evidently, WhyLabs |
| Retraining | Kubeflow Pipelines | Airflow, complex DAGs |
| Scaling | Manual replicas | HPA, VPA |
| Security | Basic | RBAC, Secrets Manager, mTLS |

## Next Steps (Post-MVP)

1. **Production MLflow**: MySQL backend, S3 artifact store
2. **Monitoring Stack**: Prometheus + Grafana
3. **Advanced Drift**: Evidently AI integration
4. **GitOps**: ArgoCD for deployments
5. **Auto-scaling**: HPA based on request volume

## Troubleshooting Quick Reference

```bash
# Pod not starting
kubectl describe pod <pod-name> -n mlops
kubectl logs <pod-name> -n mlops

# MLflow connection issues
kubectl exec -it deployment/ml-api -n mlops -- curl http://mlflow:5000/health

# Kubeflow pipeline fails
kubectl get workflows -n kubeflow
kubectl logs <workflow-pod> -n kubeflow

# Image not found in minikube
minikube image load your-image:tag
kubectl get events -n mlops

# Minikube issues
minikube status
minikube logs

# Reset everything
kubectl delete namespace mlops
kubectl delete namespace kubeflow
minikube delete
minikube start --memory=4096 --cpus=2
```
