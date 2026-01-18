# 🔧 Environment Variables Guide

## 📋 Overview

Environment variables trong Docker có 2 loại:
1. **Build-time** - Dùng khi build image (`docker build`)
2. **Runtime** - Dùng khi chạy container (`docker run` hoặc `docker-compose up`)

---

## 🏗️ Build-Time Environment Variables

### Set trong Dockerfile với `ENV`

Các biến này được **baked into image** và có sẵn trong mọi container:

```dockerfile
# Build-time ENVs (trong Dockerfile)
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${PATH}:/usr/lib/jvm/default-java/bin"
ENV PYTHONUNBUFFERED=1
```

| Variable | Value | Purpose | When Used |
|----------|-------|---------|-----------|
| `JAVA_HOME` | `/usr/lib/jvm/default-java` | GumTree cần biết Java path | Build + Runtime |
| `PATH` | `${PATH}:/usr/lib/jvm/default-java/bin` | Java commands available | Build + Runtime |
| `PYTHONUNBUFFERED` | `1` | Real-time logs (no buffering) | Build + Runtime |

### ⚠️ Build-time ARGs (không dùng trong project này)

```dockerfile
# Example (not used in our Dockerfile)
ARG BUILD_DATE
ARG VERSION
RUN echo "Building version ${VERSION} on ${BUILD_DATE}"
```

**Khác biệt ARG vs ENV:**
- `ARG`: Chỉ có trong build time, **không** có trong runtime
- `ENV`: Có trong cả build time **và** runtime

---

## 🚀 Runtime Environment Variables

### 1. Default Values (trong Dockerfile)

Các giá trị mặc định, có thể override khi chạy:

```dockerfile
# Runtime ENVs with defaults (trong Dockerfile)
ENV MLFLOW_URI=http://host.docker.internal:5000
ENV MODEL_NAME=codebuggy-detector
ENV MODEL_STAGE="Version 3"
ENV PORT=8080
```

| Variable | Default | Purpose | Override? |
|----------|---------|---------|-----------|
| `MLFLOW_URI` | `http://host.docker.internal:5000` | MLflow server URL | ✅ Yes |
| `MODEL_NAME` | `codebuggy-detector` | Model name in registry | ✅ Yes |
| `MODEL_STAGE` | `Version 3` | Model version/stage | ✅ Yes |
| `PORT` | `8080` | Flask app port | ✅ Yes |

### 2. Override trong docker-compose.yml

```yaml
services:
  codebuggy-app:
    environment:
      - MLFLOW_URI=http://mlflow:5000        # Override default
      - MODEL_NAME=codebuggy-detector         # Same as default
      - MODEL_STAGE=Version 3                 # Override default
      - PORT=8080                             # Same as default
      - PYTHONUNBUFFERED=1                    # Redundant (already in Dockerfile)
```

### 3. Override khi chạy docker run

```bash
docker run -d \
  -e MLFLOW_URI=http://my-mlflow:5000 \
  -e MODEL_STAGE="Production" \
  -e PORT=9000 \
  codebuggy-app:latest
```

---

## 📊 Complete Environment Variables Table

### Build-Time Only

| Variable | Set In | Value | Used By | Purpose |
|----------|--------|-------|---------|---------|
| `ARCH` | RUN command | `arm64` or `amd64` | Dockerfile | Detect architecture |

**Example:**
```dockerfile
RUN ARCH=$(dpkg --print-architecture) && \
    ln -sf /usr/lib/jvm/java-21-openjdk-${ARCH} /usr/lib/jvm/default-java
```

### Build-Time + Runtime

| Variable | Set In | Default Value | Used By | Purpose |
|----------|--------|---------------|---------|---------|
| `JAVA_HOME` | Dockerfile ENV | `/usr/lib/jvm/default-java` | GumTree, Java | Java installation path |
| `PATH` | Dockerfile ENV | `${PATH}:/usr/lib/jvm/default-java/bin` | System | Java commands in PATH |
| `PYTHONUNBUFFERED` | Dockerfile ENV | `1` | Python | Real-time logs |

### Runtime Only (Application)

| Variable | Set In | Default Value | Used By | Purpose | Override? |
|----------|--------|---------------|---------|---------|-----------|
| `MLFLOW_URI` | Dockerfile ENV | `http://host.docker.internal:5000` | `app.py`, `inference.py` | MLflow server URL | ✅ Yes |
| `MODEL_NAME` | Dockerfile ENV | `codebuggy-detector` | `app.py`, `inference.py` | Model name in registry | ✅ Yes |
| `MODEL_STAGE` | Dockerfile ENV | `Version 3` | `app.py`, `inference.py` | Model version/stage | ✅ Yes |
| `PORT` | Dockerfile ENV | `8080` | `app.py` | Flask server port | ✅ Yes |

---

## 🔍 How Variables Are Used

### 1. In Dockerfile (Build Time)

```dockerfile
# JAVA_HOME used during build
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Used when running download_models.py during build
RUN python download_models.py
# ^ This script can access JAVA_HOME, PYTHONUNBUFFERED, etc.
```

### 2. In Python Code (Runtime)

```python
# app.py
def initialize_predictor():
    mlflow_uri = os.getenv("MLFLOW_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "codebuggy-detector")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    
    predictor = CodeBuggyPredictor(
        mlflow_uri=mlflow_uri,
        model_name=model_name,
        model_stage=model_stage,
    )
```

### 3. In docker-compose.yml (Runtime Override)

```yaml
environment:
  - MLFLOW_URI=http://mlflow:5000  # Override Dockerfile default
  - MODEL_STAGE=Version 3           # Override Dockerfile default
```

---

## 🎯 Common Scenarios

### Scenario 1: Development (Local MLflow)

```bash
# docker-compose.yml
environment:
  - MLFLOW_URI=http://host.docker.internal:5000
  - MODEL_STAGE=Version 3
```

### Scenario 2: Production (Remote MLflow)

```bash
# docker-compose.yml or .env
environment:
  - MLFLOW_URI=https://mlflow.production.com
  - MODEL_STAGE=Production
  - MODEL_NAME=codebuggy-detector-prod
```

### Scenario 3: Testing (Different Model)

```bash
docker run -d \
  -e MLFLOW_URI=http://mlflow-test:5000 \
  -e MODEL_NAME=codebuggy-detector-test \
  -e MODEL_STAGE=Staging \
  codebuggy-app:latest
```

### Scenario 4: Custom Port

```bash
docker run -d \
  -p 9000:9000 \
  -e PORT=9000 \
  codebuggy-app:latest
```

---

## 📝 Best Practices

### ✅ DO:

1. **Set sensible defaults in Dockerfile**
   ```dockerfile
   ENV MLFLOW_URI=http://host.docker.internal:5000
   ENV MODEL_NAME=codebuggy-detector
   ```

2. **Override in docker-compose for different environments**
   ```yaml
   # docker-compose.prod.yml
   environment:
     - MLFLOW_URI=https://mlflow.prod.com
   ```

3. **Use .env file for sensitive data**
   ```bash
   # .env
   MLFLOW_URI=https://mlflow.prod.com
   MODEL_STAGE=Production
   ```
   
   ```yaml
   # docker-compose.yml
   env_file:
     - .env
   ```

4. **Document all variables**
   - In README
   - In docker-compose.yml comments
   - In this file

### ❌ DON'T:

1. **Don't hardcode secrets in Dockerfile**
   ```dockerfile
   # ❌ BAD
   ENV API_KEY=secret123
   ENV PASSWORD=admin
   ```

2. **Don't use ARG for runtime values**
   ```dockerfile
   # ❌ BAD - ARG not available at runtime
   ARG MLFLOW_URI
   # ✅ GOOD - ENV available at runtime
   ENV MLFLOW_URI=http://localhost:5000
   ```

3. **Don't override system variables unnecessarily**
   ```dockerfile
   # ❌ BAD - Can break things
   ENV PATH=/my/custom/path
   # ✅ GOOD - Append to existing
   ENV PATH="${PATH}:/my/custom/path"
   ```

---

## 🧪 Testing Environment Variables

### Test 1: Check all variables in container

```bash
docker run --rm codebuggy-app:latest env | grep -E "MLFLOW|MODEL|PORT|JAVA|PYTHON"
```

Expected output:
```
JAVA_HOME=/usr/lib/jvm/default-java
PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/jvm/default-java/bin
PYTHONUNBUFFERED=1
MLFLOW_URI=http://host.docker.internal:5000
MODEL_NAME=codebuggy-detector
MODEL_STAGE=Version 3
PORT=8080
```

### Test 2: Override at runtime

```bash
docker run --rm \
  -e MLFLOW_URI=http://test:5000 \
  -e MODEL_STAGE=Staging \
  codebuggy-app:latest \
  python -c "import os; print(f'MLFLOW_URI={os.getenv(\"MLFLOW_URI\")}'); print(f'MODEL_STAGE={os.getenv(\"MODEL_STAGE\")}')"
```

Expected output:
```
MLFLOW_URI=http://test:5000
MODEL_STAGE=Staging
```

### Test 3: Check in running container

```bash
docker-compose up -d
docker-compose exec codebuggy-app env | grep MLFLOW
```

---

## 🔄 Environment Variable Precedence

**Order (highest to lowest priority):**

1. **docker run -e** (command line)
   ```bash
   docker run -e MLFLOW_URI=http://override:5000 ...
   ```

2. **docker-compose.yml environment**
   ```yaml
   environment:
     - MLFLOW_URI=http://compose:5000
   ```

3. **docker-compose.yml env_file**
   ```yaml
   env_file:
     - .env
   ```

4. **Dockerfile ENV**
   ```dockerfile
   ENV MLFLOW_URI=http://default:5000
   ```

**Example:**
```dockerfile
# Dockerfile
ENV PORT=8080
```

```yaml
# docker-compose.yml
environment:
  - PORT=9000  # This overrides Dockerfile
```

```bash
# Command line
docker-compose run -e PORT=7000 codebuggy-app  # This overrides both
# Final value: PORT=7000
```

---

## 📚 Summary

### Build-Time Variables:
- `JAVA_HOME` - Java installation path
- `PATH` - System PATH with Java
- `PYTHONUNBUFFERED` - Python logging
- `ARCH` (temporary) - Architecture detection

### Runtime Variables (Can Override):
- `MLFLOW_URI` - MLflow server URL
- `MODEL_NAME` - Model name in registry
- `MODEL_STAGE` - Model version/stage
- `PORT` - Flask server port

### How to Override:
1. **docker-compose.yml**: Best for different environments
2. **docker run -e**: Best for quick tests
3. **.env file**: Best for sensitive data

---

## 🎯 Quick Reference

```bash
# View all variables
docker run --rm codebuggy-app:latest env

# Override single variable
docker run -e MODEL_STAGE=Staging codebuggy-app:latest

# Override multiple variables
docker run \
  -e MLFLOW_URI=http://test:5000 \
  -e MODEL_STAGE=Staging \
  -e PORT=9000 \
  codebuggy-app:latest

# Use with docker-compose
docker-compose up -d

# Check variables in running container
docker-compose exec codebuggy-app env
```

---

**Last Updated**: 2026-01-18  
**Version**: 1.0
