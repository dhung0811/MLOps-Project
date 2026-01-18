#!/bin/bash
# Script to build and test CodeBuggy Docker image

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐳 CodeBuggy Docker Build Script${NC}"
echo "=================================="

# Parse arguments
BUILD_TYPE=${1:-"standard"}  # standard, optimized, or compose

case $BUILD_TYPE in
  "standard")
    echo -e "${YELLOW}Building standard Docker image...${NC}"
    docker build -t codebuggy-app:latest .
    echo -e "${GREEN}✓ Build complete!${NC}"
    echo ""
    echo "To run:"
    echo "  docker run -d -p 8080:8080 --name codebuggy-app codebuggy-app:latest"
    ;;
    
  "optimized")
    echo -e "${YELLOW}Building optimized Docker image (multi-stage)...${NC}"
    docker build -f Dockerfile.optimized -t codebuggy-app:optimized .
    echo -e "${GREEN}✓ Build complete!${NC}"
    echo ""
    echo "Image sizes:"
    docker images | grep codebuggy-app
    echo ""
    echo "To run:"
    echo "  docker run -d -p 8080:8080 --name codebuggy-app codebuggy-app:optimized"
    ;;
    
  "compose")
    echo -e "${YELLOW}Building with Docker Compose...${NC}"
    docker-compose build
    echo -e "${GREEN}✓ Build complete!${NC}"
    echo ""
    echo "To run:"
    echo "  docker-compose up -d"
    ;;
    
  "test")
    echo -e "${YELLOW}Building and testing...${NC}"
    
    # Build
    docker build -t codebuggy-app:test .
    
    # Run in background
    echo "Starting container..."
    docker run -d \
      --name codebuggy-test \
      -p 8080:8080 \
      -e MLFLOW_URI=http://host.docker.internal:5000 \
      codebuggy-app:test
    
    # Wait for startup
    echo "Waiting for app to start..."
    sleep 10
    
    # Test health endpoint
    echo "Testing health endpoint..."
    if curl -f http://localhost:8080/health; then
      echo -e "${GREEN}✓ Health check passed!${NC}"
    else
      echo -e "${RED}✗ Health check failed!${NC}"
      docker logs codebuggy-test
      docker stop codebuggy-test
      docker rm codebuggy-test
      exit 1
    fi
    
    # Show logs
    echo ""
    echo "Container logs:"
    docker logs codebuggy-test
    
    # Cleanup
    echo ""
    echo "Cleaning up..."
    docker stop codebuggy-test
    docker rm codebuggy-test
    
    echo -e "${GREEN}✓ Test complete!${NC}"
    ;;
    
  *)
    echo -e "${RED}Unknown build type: $BUILD_TYPE${NC}"
    echo ""
    echo "Usage: $0 [standard|optimized|compose|test]"
    echo ""
    echo "Options:"
    echo "  standard   - Build standard Docker image (default)"
    echo "  optimized  - Build optimized multi-stage image"
    echo "  compose    - Build with Docker Compose"
    echo "  test       - Build and run quick test"
    exit 1
    ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
