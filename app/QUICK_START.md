# 🚀 Quick Start - CodeBuggy Docker

## One-Line Start

```bash
cd /Users/hungnguyen/dacn/app && docker-compose up -d
```

Then open: http://localhost:8080

---

## Common Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f

# Rebuild
docker-compose up -d --build

# Health check
curl http://localhost:8080/health

# Test inference
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "buggy_code": "public int sum(int[] arr) { int s = 0; for (int i = 0; i <= arr.length; i++) { s += arr[i]; } return s; }",
    "fixed_code": "public int sum(int[] arr) { int s = 0; for (int i = 0; i < arr.length; i++) { s += arr[i]; } return s; }"
  }'
```

---

## Environment Variables

### Quick Override:
```bash
# Change MLflow URL
docker run -e MLFLOW_URI=http://my-mlflow:5000 codebuggy-app

# Change model stage
docker run -e MODEL_STAGE=Production codebuggy-app

# Multiple overrides
docker run \
  -e MLFLOW_URI=http://test:5000 \
  -e MODEL_STAGE=Staging \
  codebuggy-app
```

### Check Variables:
```bash
docker-compose exec codebuggy-app env | grep -E "MLFLOW|MODEL|PORT"
```

See `ENV_QUICK_REF.md` for all variables.

---

## Troubleshooting

### Port already in use
```bash
lsof -i :8080
kill -9 <PID>
```

### Model not found
Check MLflow: http://localhost:5000

### Container won't start
```bash
docker-compose logs codebuggy-app
```

---

## Documentation

- **Quick Start**: This file
- **Environment Variables**: `ENV_QUICK_REF.md`
- **Full Guide**: `DOCKER_GUIDE.md`
- **Changes**: `DOCKER_CHANGES_SUMMARY.md`
- **Build Status**: `BUILD_SUCCESS.md`

---

**That's it! 🎉**
