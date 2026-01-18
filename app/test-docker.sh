#!/bin/bash
# Quick test script for Docker build

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🧪 Testing CodeBuggy Docker Build${NC}"
echo "=================================="
echo ""

# Test 1: Build
echo -e "${YELLOW}Test 1: Building image...${NC}"
if docker build -t codebuggy-test:latest -f Dockerfile . > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi
echo ""

# Test 2: Java
echo -e "${YELLOW}Test 2: Checking Java...${NC}"
JAVA_VERSION=$(docker run --rm codebuggy-test:latest java -version 2>&1 | head -1)
if [[ $JAVA_VERSION == *"21"* ]]; then
    echo -e "${GREEN}✅ Java 21 installed${NC}"
    echo "   $JAVA_VERSION"
else
    echo -e "${RED}❌ Java version incorrect${NC}"
    exit 1
fi
echo ""

# Test 3: JAVA_HOME
echo -e "${YELLOW}Test 3: Checking JAVA_HOME...${NC}"
JAVA_HOME=$(docker run --rm codebuggy-test:latest bash -c 'echo $JAVA_HOME')
if [[ -n "$JAVA_HOME" ]]; then
    echo -e "${GREEN}✅ JAVA_HOME set: $JAVA_HOME${NC}"
else
    echo -e "${RED}❌ JAVA_HOME not set${NC}"
    exit 1
fi
echo ""

# Test 4: GumTree
echo -e "${YELLOW}Test 4: Checking GumTree...${NC}"
if docker run --rm codebuggy-test:latest resources/gumtree-4.0.0-beta4/bin/gumtree list GENERATORS > /dev/null 2>&1; then
    echo -e "${GREEN}✅ GumTree works${NC}"
else
    echo -e "${RED}❌ GumTree failed${NC}"
    exit 1
fi
echo ""

# Test 5: Python packages
echo -e "${YELLOW}Test 5: Checking Python packages...${NC}"
TORCH_VERSION=$(docker run --rm codebuggy-test:latest python -c "import torch; print(torch.__version__)")
TRANSFORMERS_VERSION=$(docker run --rm codebuggy-test:latest python -c "import transformers; print(transformers.__version__)")
echo -e "${GREEN}✅ torch: $TORCH_VERSION${NC}"
echo -e "${GREEN}✅ transformers: $TRANSFORMERS_VERSION${NC}"
echo ""

# Test 6: Model pre-downloaded
echo -e "${YELLOW}Test 6: Checking pre-downloaded models...${NC}"
if docker run --rm codebuggy-test:latest ls -la /root/.cache/huggingface/hub/ | grep -q "graphcodebert"; then
    echo -e "${GREEN}✅ Models pre-downloaded${NC}"
else
    echo -e "${YELLOW}⚠️  Models may not be cached (will download at runtime)${NC}"
fi
echo ""

# Summary
echo "=================================="
echo -e "${GREEN}🎉 All tests passed!${NC}"
echo ""
echo "Next steps:"
echo "  1. docker-compose up -d"
echo "  2. curl http://localhost:8080/health"
echo "  3. Open http://localhost:8080"
echo ""
