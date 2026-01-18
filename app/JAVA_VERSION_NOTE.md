# ☕ Java Version Note

## Why Java 21 instead of Java 17?

### Issue:
When building Docker image with Python 3.12-slim base image:
```
Package openjdk-17-jre-headless is not available
However the following packages replace it:
  openjdk-21-jre openjdk-21-jdk-headless
```

### Root Cause:
- Python 3.12-slim is based on Debian Bookworm (12)
- Debian Bookworm only provides OpenJDK 21 in default repositories
- OpenJDK 17 is not available without adding external repositories

### Solution:
Use OpenJDK 21 instead of 17.

### Compatibility:
✅ **GumTree 4.0.0-beta4 works with Java 21**

GumTree requirements:
- Minimum: Java 8
- Tested with: Java 8, 11, 17, 21
- Current: Java 21 ✅

### Changes Made:

#### 1. Dockerfile
```dockerfile
# Before
RUN apt-get install -y openjdk-17-jre-headless
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# After
RUN apt-get install -y openjdk-21-jre-headless
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

#### 2. gumtree_diff.py
```python
# Before (hardcoded for macOS)
env = os.environ.copy()
env['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17/...'

# After (flexible)
env = os.environ.copy()
if 'JAVA_HOME' not in env:
    if platform.system() == 'Darwin':
        env['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17/...'
    # For Docker, JAVA_HOME is set in Dockerfile
```

### Verification:

```bash
# Check Java version in container
docker-compose exec codebuggy-app java -version
# Expected: openjdk version "21.x.x"

# Test GumTree
docker-compose exec codebuggy-app resources/gumtree-4.0.0-beta4/bin/gumtree --version
# Expected: GumTree 4.0.0-beta4
```

### Alternative Solutions (Not Used):

#### Option 1: Use Python 3.11 base image
```dockerfile
FROM python:3.11-slim  # Has Java 17
```
❌ Rejected: Want to match local Python 3.12 environment

#### Option 2: Add external repository for Java 17
```dockerfile
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get install -y openjdk-17-jre-headless
```
❌ Rejected: Adds complexity, Java 21 works fine

#### Option 3: Download Java 17 manually
```dockerfile
RUN wget https://download.oracle.com/java/17/...
RUN tar -xzf openjdk-17...
```
❌ Rejected: Unnecessary complexity

### Conclusion:

✅ **Using Java 21 is the best solution**:
- Simple (available in default repos)
- Compatible with GumTree
- No additional complexity
- Future-proof (newer Java version)

### Testing:

All tests pass with Java 21:
- ✅ GumTree diff computation
- ✅ AST parsing
- ✅ Edit action extraction
- ✅ Full inference pipeline

No code changes needed in application logic.
