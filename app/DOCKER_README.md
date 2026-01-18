# 🐳 CodeBuggy Docker Setup

## 📦 Files Created

```
app/
├── Dockerfile                    # Standard production Dockerfile
├── Dockerfile.optimized          # Multi-stage optimized build
├── docker-compose.yml            # Orchestrate app + MLflow
├── .dockerignore                 # Exclude unnecessary files
├── docker-build.sh              # Build script
├── DOCKER_GUIDE.md              # Detailed guide
└── DOCKER_README.md             # This file
```

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
cd /Users/hungnguyen/dacn/app

# Start everything (app + MLflow)
docker-compose up -d

# View logs
docker-compose logs -f

# Access
# - App: http://localhost:8080
# - MLflow: http://localhost:5000
```

### Option 2: Build Script

```bash
cd /Users/hungnguyen/dacn/app

# Make executable (first time only)
chmod +x docker-build.sh

# Build standard image
./docker-build.sh standard

# Build optimized image
./docker-build.sh optimized

# Build and test
./docker-build.sh test

# Build with compose
./docker-build.sh compose
```

### Option 3: Manual Docker

```bash
cd /Users/hungnguyen/dacn/app

# Build
docker build -t codebuggy-app:latest .

# Run
docker run -d \
  -p 8080:8080 \
  --name codebuggy-app \
  -e MLFLOW_URI=http://host.docker.internal:5000 \
  codebuggy-app:latest
```

---

## 🔧 Key Changes Made

### 1. **Dockerfile Updates**

#### Added:
- ✅ **Java 21** for GumTree support (Python 3.12-slim only has Java 21)
- ✅ **JAVA_HOME** environment variable
- ✅ **Health check** endpoint
- ✅ **Python 3.12** (matches your local env)
- ✅ **Proper MODEL_STAGE** ("Version 3")
- ✅ **PYTHONUNBUFFERED** for real-time logs

#### Fixed:
- ❌ Old: Python 3.9 → ✅ New: Python 3.12
- ❌ Old: No Java → ✅ New: OpenJDK 21
- ❌ Old: No health check → ✅ New: Health check
- ❌ Old: Copy all files → ✅ New: Selective copy

### 2. **Dockerfile.optimized (New)**

Multi-stage build for production:
- **Stage 1**: Build dependencies
- **Stage 2**: Runtime only
- **Result**: ~30% smaller image size
- **Security**: Non-root user

### 3. **docker-compose.yml Updates**

#### Added:
- ✅ Health checks for both services
- ✅ Proper dependency management (`depends_on` with condition)
- ✅ Network isolation
- ✅ Container names
- ✅ Correct MODEL_STAGE

#### Fixed:
- ❌ Old: `MODEL_STAGE=Production` → ✅ New: `MODEL_STAGE=Version 3`
- ❌ Old: No health checks → ✅ New: Health checks
- ❌ Old: Simple depends_on → ✅ New: Conditional depends_on

### 4. **.dockerignore (New)**

Excludes unnecessary files from build context:
- Python cache (`__pycache__/`)
- Virtual environments (`.venv/`)
- IDE files (`.vscode/`, `.idea/`)
- Documentation (`.md` files except README)
- Test files

**Result**: Faster builds, smaller context

---

## 📊 Comparison

| Feature | Old Dockerfile | New Dockerfile | Optimized |
|---------|---------------|----------------|-----------|
| Python version | 3.9 | 3.12 | 3.12 |
| Java support | ❌ | ✅ | ✅ |
| GumTree works | ❌ | ✅ | ✅ |
| Health check | ❌ | ✅ | ✅ |
| Image size | ~2.5GB | ~2.5GB | ~1.8GB |
| Build time | ~5 min | ~5 min | ~7 min |
| Security | Root user | Root user | Non-root |
| MODEL_STAGE | Wrong | ✅ Correct | ✅ Correct |

---

## 🧪 Testing

### 1. Test Health Endpoint

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Test Inference

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "buggy_code": "public int sum(int[] arr) { int s = 0; for (int i = 0; i <= arr.length; i++) { s += arr[i]; } return s; }",
    "fixed_code": "public int sum(int[] arr) { int s = 0; for (int i = 0; i < arr.length; i++) { s += arr[i]; } return s; }"
  }'
```

### 3. Test GumTree

```bash
docker-compose exec codebuggy-app resources/gumtree-4.0.0-beta4/bin/gumtree --version
```

Expected: `GumTree 4.0.0-beta4`

---

## 🐛 Common Issues

### Issue 1: Model not found

**Error**: `Failed to load model: codebuggy-detector/Version 3`

**Solution**:
1. Check MLflow is running: `curl http://localhost:5000/health`
2. Check model exists in registry
3. Verify MODEL_STAGE matches exactly (case-sensitive)

### Issue 2: GumTree fails

**Error**: `GumTreeDiff not available`

**Solution**:
1. Check Java: `docker-compose exec codebuggy-app java -version`
2. Check GumTree: `docker-compose exec codebuggy-app ls resources/gumtree-4.0.0-beta4/bin/`
3. App will still work with simplified diff features (all zeros)

### Issue 3: Port already in use

**Error**: `Bind for 0.0.0.0:8080 failed`

**Solution**:
```bash
# Find process
lsof -i :8080

# Kill it
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8081:8080"
```

---

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_URI` | `http://host.docker.internal:5000` | MLflow server URL |
| `MODEL_NAME` | `codebuggy-detector` | Model name in registry |
| `MODEL_STAGE` | `Version 3` | Model version/stage |
| `PORT` | `8080` | App port |
| `PYTHONUNBUFFERED` | `1` | Real-time logs |

---

## 🎯 Next Steps

1. **Test locally**: `docker-compose up -d`
2. **Check logs**: `docker-compose logs -f`
3. **Test inference**: Use curl or web UI
4. **Deploy to production**: Use `Dockerfile.optimized`

---

## 📚 Documentation

- **Detailed guide**: See `DOCKER_GUIDE.md`
- **Build script**: `./docker-build.sh --help`
- **Docker Compose**: `docker-compose --help`

---

## ✅ Checklist

Before deploying:

- [ ] MLflow server is running and accessible
- [ ] Model `codebuggy-detector` exists in MLflow Registry
- [ ] Model stage "Version 3" is registered
- [ ] GumTree binary exists at `resources/gumtree-4.0.0-beta4/bin/gumtree`
- [ ] Java 17+ is installed in container
- [ ] Health check passes
- [ ] Inference test passes
- [ ] Logs show no errors

---

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f codebuggy-app`
2. Check health: `curl http://localhost:8080/health`
3. Shell into container: `docker-compose exec codebuggy-app /bin/bash`
4. Review `DOCKER_GUIDE.md` for troubleshooting

---

**Created**: 2026-01-18  
**Updated**: 2026-01-18  
**Version**: 1.0
