# 🔧 Meta Tensor Fix

## Problem

```
NotImplementedError: Cannot copy out of meta tensor; no data! 
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() 
when moving module from meta to a different device.
```

This error occurred **only in Docker**, not on host machine.

---

## Root Cause

### Issue 1: Unpinned Package Versions
```txt
# Old codebuggy_requirements_docker.txt
transformers    # ❌ No version - installs latest
torch           # ❌ No version - installs latest
```

Different versions between host and Docker caused compatibility issues.

### Issue 2: Model Loading Method
```python
# Old code - problematic in Docker
self.encoder = AutoModel.from_pretrained(model_name).to(device)
```

When `transformers` uses `low_cpu_mem_usage=True` by default (in newer versions), it loads models to "meta" device first, causing the error when moving to actual device.

---

## Solution

### Fix 1: Pin Package Versions ✅

```txt
# New codebuggy_requirements_docker.txt
transformers==4.57.5  # ✅ Match host version
torch==2.9.1          # ✅ Match host version
mlflow==2.9.2         # ✅ Pin version
```

### Fix 2: Explicit Model Loading ✅

```python
# New code - works in Docker
self.encoder = AutoModel.from_pretrained(
    graphcodebert_model,
    torch_dtype=torch.float32,  # Explicit dtype
    low_cpu_mem_usage=False,     # Disable meta device
)
self.encoder = self.encoder.to(self.device)
```

### Fix 3: Pre-download Models ✅

Created `download_models.py` to download models during Docker build:

```python
# download_models.py
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
)
```

Added to Dockerfile:
```dockerfile
COPY download_models.py ./
RUN python download_models.py  # Pre-download during build
```

**Benefits**:
- Faster container startup
- Verify model loading works during build
- Avoid runtime download issues

---

## Verification

### Before Fix ❌
```bash
$ docker run codebuggy-app
NotImplementedError: Cannot copy out of meta tensor
```

### After Fix ✅
```bash
$ docker run codebuggy-app
Device: cpu
Loading microsoft/graphcodebert-base...
✓ GraphCodeBERT loaded
Loading model from MLflow...
✓ Model loaded successfully
```

---

## Files Changed

1. ✅ `inference.py` - Updated model loading
2. ✅ `codebuggy_requirements_docker.txt` - Pinned versions
3. ✅ `download_models.py` - New pre-download script
4. ✅ `Dockerfile` - Added model pre-download step
5. ✅ `Dockerfile.optimized` - Added model pre-download step

---

## Why This Happened

### Host vs Docker Differences

| Aspect | Host | Docker (before fix) |
|--------|------|---------------------|
| transformers | 4.57.5 | Latest (4.58+) |
| torch | 2.9.1 | Latest (2.9.2+) |
| low_cpu_mem_usage | False (default in 4.57) | True (default in 4.58+) |
| Result | ✅ Works | ❌ Meta tensor error |

### Transformers Version Changes

- **v4.57.x**: `low_cpu_mem_usage=False` by default
- **v4.58+**: `low_cpu_mem_usage=True` by default (uses meta device)

When `low_cpu_mem_usage=True`:
1. Model loads to "meta" device (no actual data)
2. `.to(device)` tries to copy from meta → real device
3. ❌ Error: "Cannot copy out of meta tensor"

---

## Best Practices

### ✅ DO:
- Pin all package versions in requirements
- Use explicit `torch_dtype` and `low_cpu_mem_usage`
- Pre-download models during build
- Test in Docker before deploying

### ❌ DON'T:
- Use unpinned versions (`transformers` instead of `transformers==4.57.5`)
- Chain `.from_pretrained().to(device)` without explicit params
- Assume host and Docker have same package versions
- Download models at runtime (slow startup)

---

## Testing

### Test 1: Model Loading
```bash
docker run --rm codebuggy-app:fixed python -c "
from inference import CodeBuggyPredictor
p = CodeBuggyPredictor(mlflow_uri='http://localhost:5000')
print('✓ Success!')
"
```

### Test 2: Full Inference
```bash
docker-compose up -d
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"buggy_code": "...", "fixed_code": "..."}'
```

---

## Related Issues

- PyTorch Issue: https://github.com/pytorch/pytorch/issues/92379
- Transformers Issue: https://github.com/huggingface/transformers/issues/25633
- MLflow compatibility with newer transformers

---

## Summary

✅ **Fixed by**:
1. Pinning package versions to match host
2. Explicit model loading parameters
3. Pre-downloading models during build

✅ **Result**: Container works identically to host machine

**Date**: 2026-01-18  
**Status**: ✅ RESOLVED
