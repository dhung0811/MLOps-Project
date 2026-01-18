# ⚡ Environment Variables - Quick Reference

## 📋 All Variables

| Variable | Type | Default | Override? | Used By |
|----------|------|---------|-----------|---------|
| `JAVA_HOME` | Build+Runtime | `/usr/lib/jvm/default-java` | ❌ No | GumTree, Java |
| `PATH` | Build+Runtime | `${PATH}:/usr/lib/jvm/.../bin` | ❌ No | System |
| `PYTHONUNBUFFERED` | Build+Runtime | `1` | ❌ No | Python |
| `MLFLOW_URI` | Runtime | `http://host.docker.internal:5000` | ✅ Yes | app.py |
| `MODEL_NAME` | Runtime | `codebuggy-detector` | ✅ Yes | app.py |
| `MODEL_STAGE` | Runtime | `Version 3` | ✅ Yes | app.py |
| `PORT` | Runtime | `8080` | ✅ Yes | Flask |

---

## 🔧 How to Override

### Method 1: docker-compose.yml (Recommended)
```yaml
environment:
  - MLFLOW_URI=http://mlflow:5000
  - MODEL_STAGE=Production
```

### Method 2: docker run
```bash
docker run -e MLFLOW_URI=http://test:5000 codebuggy-app
```

### Method 3: .env file
```bash
# .env
MLFLOW_URI=http://mlflow:5000
MODEL_STAGE=Production
```

```yaml
# docker-compose.yml
env_file:
  - .env
```

---

## 🎯 Common Use Cases

### Development
```yaml
environment:
  - MLFLOW_URI=http://host.docker.internal:5000
  - MODEL_STAGE=Version 3
```

### Production
```yaml
environment:
  - MLFLOW_URI=https://mlflow.prod.com
  - MODEL_STAGE=Production
```

### Testing
```bash
docker run \
  -e MLFLOW_URI=http://mlflow-test:5000 \
  -e MODEL_STAGE=Staging \
  codebuggy-app
```

---

## 🔍 Check Variables

```bash
# In running container
docker-compose exec codebuggy-app env | grep -E "MLFLOW|MODEL|PORT"

# Before starting
docker run --rm codebuggy-app env | grep -E "MLFLOW|MODEL|PORT"
```

---

## ⚠️ Important Notes

- ❌ **Cannot override**: `JAVA_HOME`, `PATH`, `PYTHONUNBUFFERED` (need rebuild)
- ✅ **Can override**: `MLFLOW_URI`, `MODEL_NAME`, `MODEL_STAGE`, `PORT`
- 🔄 **Precedence**: `docker run -e` > `docker-compose` > `Dockerfile`

---

## 📚 Full Documentation

See `ENVIRONMENT_VARIABLES.md` for complete details.
