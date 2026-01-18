# Use Python 3.12 to match your local environment
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - gcc/g++ for compiling Python packages (numpy, torch, etc.)
# - git for potential package installations
# - openjdk-21-jre-headless for GumTree (Java-based AST diff tool)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for GumTree (auto-detect architecture)
# This works for both amd64 and arm64
RUN ARCH=$(dpkg --print-architecture) && \
    ln -sf /usr/lib/jvm/java-21-openjdk-${ARCH} /usr/lib/jvm/default-java

# Set environment variable
ENV JAVA_HOME=/usr/lib/jvm/default-java \
    PATH="${PATH}:/usr/lib/jvm/default-java/bin"

# Upgrade pip to ensure support for newer packages
RUN python -m pip install --upgrade pip

# Copy requirement files first to leverage Docker cache
COPY app_requirements_docker.txt codebuggy_requirements_docker.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r app_requirements_docker.txt

# Copy application code
COPY app.py inference.py examples.json download_models.py ./
COPY utils/ ./utils/
COPY templates/ ./templates/
COPY static/ ./static/
COPY resources/ ./resources/

# Create output directory if needed
RUN mkdir -p output

# Pre-download models during build to avoid runtime issues
RUN python download_models.py

# Make GumTree executable
RUN chmod +x resources/gumtree-4.0.0-beta4/bin/gumtree

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Environment variables
ENV MLFLOW_URI=http://host.docker.internal:5000
ENV MODEL_NAME=codebuggy-detector
ENV MODEL_STAGE="Version 3"
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run app.py when the container launches
CMD ["python", "app.py"]
