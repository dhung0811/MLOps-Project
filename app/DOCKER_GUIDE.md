# 🐳 Docker Guide - CodeBuggy Web App

## 📋 Tổng quan

Có 3 cách để chạy CodeBuggy app với Docker:

1. **Docker Compose** (Khuyến nghị) - Chạy cả app + MLflow
2. **Docker standalone** - Chỉ chạy app, MLflow chạy riêng
3. **Docker optimized** - Production build với multi-stage

---

## 🚀 Cách 1: Docker Compose (Khuyến nghị)

### Build và chạy:

```bash
cd /Users/hungnguyen/dacn/app

# Build và start tất cả services
docker-compose up --build

# Hoặc chạy background
docker-compose up -d --build

# Xem logs
docker-compose logs -f codebuggy-app
docker-compose logs -f mlflow

# Stop
docker-compose down

# Stop và xóa volumes
docker-compose down -v
```

### Truy cập:

- **Web App**: http://localhost:8080
- **MLflow UI**: http://localhost:5000
- **Health Check**: http://localhost:8080/health

---

## 🔧 Cách 2: Docker Standalone

### Chạy MLflow trước:

```bash
# Start MLflow container
docker run -d \
  --name mlflow-server \
  -p 5000:5000 \
  -v mlflow-data:/mlflow \
  ghcr.io/mlflow/mlflow:v2.9.2 \
  mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root /mlflow/artifacts
```

### Build và chạy app:

```bash
cd /Users/hungnguyen/dacn/app

# Build image
docker build -t codebuggy-app:latest .

# Run container
docker run -d \
  --name codebuggy-app \
  -p 8080:8080 \
  -e MLFLOW_URI=http://host.docker.internal:5000 \
  -e MODEL_NAME=codebuggy-detector \
  -e MODEL_STAGE="Version 3" \
  -v $(pwd)/output:/app/output \
  codebuggy-app:latest

# Xem logs
docker logs -f codebuggy-app

# Stop
docker stop codebuggy-app
docker rm codebuggy-app
```

---

## ⚡ Cách 3: Docker Optimized (Production)

### Build optimized image:

```bash
cd /Users/hungnguyen/dacn/app

# Build với multi-stage
docker build -f Dockerfile.optimized -t codebuggy-app:optimized .

# Run
docker run -d \
  --name codebuggy-app-prod \
  -p 8080:8080 \
  -e MLFLOW_URI=http://host.docker.internal:5000 \
  -e MODEL_NAME=codebuggy-detector \
  -e MODEL_STAGE="Version 3" \
  codebuggy-app:optimized
```

### So sánh kích thước:

```bash
docker images | grep codebuggy-app
# codebuggy-app:latest     ~2.5GB
# codebuggy-app:optimized  ~1.8GB (nhỏ hơn ~30%)
```

---

## 🔍 Troubleshooting

### 1. Model không load được

**Lỗi**: `Failed to load model from MLflow`

**Giải pháp**:

```bash
# Kiểm tra MLflow có chạy không
curl http://localhost:5000/health

# Kiểm tra model có trong registry không
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=codebuggy-detector

# Nếu chạy trong Docker, dùng network name
docker-compose exec codebuggy-app curl http://mlflow:5000/health
```

### 2. GumTree không hoạt động

**Lỗi**: `GumTree not found` hoặc `Java not found`

**Giải pháp**:

```bash
# Kiểm tra Java trong container (cần Java 8+, có Java 21)
docker-compose exec codebuggy-app java -version

# Kiểm tra GumTree
docker-compose exec codebuggy-app ls -la resources/gumtree-4.0.0-beta4/bin/

# Test GumTree
docker-compose exec codebuggy-app resources/gumtree-4.0.0-beta4/bin/gumtree --version
```

**Note**: GumTree 4.0.0-beta4 tương thích với Java 8-21.

### 3. Port đã được sử dụng

**Lỗi**: `Bind for 0.0.0.0:8080 failed: port is already allocated`

**Giải pháp**:

```bash
# Tìm process đang dùng port
lsof -i :8080

# Kill process
kill -9 <PID>

# Hoặc đổi port trong docker-compose.yml
ports:
  - "8081:8080"  # Dùng port 8081 thay vì 8080
```

### 4. Container crash ngay sau khi start

**Giải pháp**:

```bash
# Xem logs chi tiết
docker-compose logs codebuggy-app

# Chạy interactive để debug
docker-compose run --rm codebuggy-app /bin/bash

# Trong container, test manual
python app.py
```

### 5. Thiếu dependencies

**Lỗi**: `ModuleNotFoundError: No module named 'xxx'`

**Giải pháp**:

```bash
# Rebuild image với --no-cache
docker-compose build --no-cache

# Hoặc
docker build --no-cache -t codebuggy-app:latest .
```

---

## 📊 Monitoring

### Health checks:

```bash
# App health
curl http://localhost:8080/health

# MLflow health
curl http://localhost:5000/health

# Docker health status
docker ps
# Xem cột STATUS, nên thấy "healthy"
```

### Resource usage:

```bash
# Xem CPU/Memory usage
docker stats

# Xem logs realtime
docker-compose logs -f --tail=100
```

---

## 🧪 Testing

### Test inference API:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "buggy_code": "public int sum(int[] arr) { int s = 0; for (int i = 0; i <= arr.length; i++) { s += arr[i]; } return s; }",
    "fixed_code": "public int sum(int[] arr) { int s = 0; for (int i = 0; i < arr.length; i++) { s += arr[i]; } return s; }"
  }'
```

### Test với Python:

```python
import requests

response = requests.post(
    "http://localhost:8080/predict",
    json={
        "buggy_code": "public int sum(int[] arr) { ... }",
        "fixed_code": "public int sum(int[] arr) { ... }"
    }
)

print(response.json())
```

---

## 🔐 Production Deployment

### Environment variables:

```bash
# .env file
MLFLOW_URI=http://mlflow-prod.example.com
MODEL_NAME=codebuggy-detector
MODEL_STAGE=Production
PORT=8080
WORKERS=4
```

### Run with .env:

```bash
docker-compose --env-file .env up -d
```

### Security best practices:

1. ✅ Chạy với non-root user (đã có trong Dockerfile.optimized)
2. ✅ Sử dụng health checks
3. ✅ Limit resources:

```yaml
services:
  codebuggy-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## 📝 Notes

### Về GumTree:

- Cần Java 17+ để chạy
- Path: `resources/gumtree-4.0.0-beta4/bin/gumtree`
- Nếu không có GumTree, app vẫn chạy nhưng diff features sẽ là all-zero

### Về MLflow:

- Model phải được register trước trong MLflow Registry
- Stage name phải khớp chính xác (case-sensitive)
- Default stage: "Version 3"

### Về Resources:

- GraphCodeBERT model: ~500MB
- PyTorch model: ~100MB
- Total RAM cần: ~2-4GB
- Lần đầu chạy sẽ download models, mất ~5-10 phút

---

## 🎯 Quick Commands

```bash
# Start everything
docker-compose up -d

# Rebuild after code changes
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Clean everything (including volumes)
docker-compose down -v
docker system prune -a

# Shell into container
docker-compose exec codebuggy-app /bin/bash

# Check health
curl http://localhost:8080/health
```
