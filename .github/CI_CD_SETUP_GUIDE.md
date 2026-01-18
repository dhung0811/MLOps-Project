# CI/CD Setup Guide

Complete guide to setting up GitHub Actions CI/CD pipeline for CodeBuggy project.

## Overview

Single CI/CD workflow that builds Docker images and updates GitOps repository automatically.

## Prerequisites

- Docker Hub account
- GitOps repository (separate repo for Kubernetes manifests)
- GitHub repository with admin access

## Step 1: Create Docker Hub Access Token

1. Go to [Docker Hub](https://hub.docker.com/)
2. Navigate to Account Settings → Security
3. Click "New Access Token"
4. Name: `github-actions-codebuggy`
5. Permissions: Read, Write, Delete
6. Copy the token (you won't see it again!)

## Step 2: Create GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Name: `gitops-codebuggy`
4. Scopes: `repo` (full control)
5. Copy the token

## Step 3: Set Up GitOps Repository

Create a separate repository for Kubernetes manifests:

```bash
# Create new repository on GitHub (e.g., dhung04/codebuggy-gitops)
git clone https://github.com/dhung04/codebuggy-gitops.git
cd codebuggy-gitops
```

See [GITOPS_TEMPLATE.md](./GITOPS_TEMPLATE.md) for recommended structure.

## Step 4: Configure GitHub Secrets

In your main repository:

1. Go to Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add the following secrets:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `DOCKER_PASSWORD` | `<docker-hub-token>` | Docker Hub access token |
| `GITOPS_REPO` | `dhung04/codebuggy-gitops` | GitOps repository (owner/repo) |
| `GITOPS_TOKEN` | `<github-token>` | GitHub token for GitOps repo |

## Step 5: Verify Workflow Configuration

Check `.github/workflows/ci-cd.yaml`:

```yaml
env:
  DOCKER_REGISTRY: docker.io
  DOCKER_USERNAME: dhung04  # ← Update to your Docker Hub username
  IMAGE_NAME: codebuggy
```

## Step 6: Test the Pipeline

### Option 1: Push to Main Branch

```bash
git add .
git commit -m "test: trigger CI/CD pipeline"
git push origin main
```

### Option 2: Manual Trigger

1. Go to Actions tab in GitHub
2. Select "CI/CD Pipeline"
3. Click "Run workflow"
4. Select branch: `main`
5. Click "Run workflow"

## Step 7: Monitor Pipeline

1. Go to Actions tab
2. Click on the running workflow
3. Monitor each step:
   - ✅ Build and push Docker image (multi-arch: amd64, arm64)
   - ✅ Update GitOps repository
   - ✅ Summary

## Step 8: Verify Results

### Check Docker Hub

```bash
docker pull dhung04/codebuggy:latest
docker images | grep codebuggy
```

### Check GitOps Repository

```bash
cd codebuggy-gitops
git pull
cat kustomization.yaml  # Check newTag
cat deployment.yaml     # Check image tag
```

### Deploy to Kubernetes

```bash
kubectl apply -k .
kubectl get pods -n codebuggy
```

## Troubleshooting

### Build Fails

- Check Docker Hub credentials in GitHub Secrets
- Verify Dockerfile syntax
- Check build logs in Actions tab

### GitOps Update Fails

- Verify `GITOPS_TOKEN` has repo access
- Check `GITOPS_REPO` format (owner/repo)
- Ensure GitOps repo exists and is accessible

### Deployment Fails

- Check Kubernetes cluster connectivity
- Verify namespace exists: `kubectl get ns codebuggy`
- Check resource limits and pod status

## Security Best Practices

1. **Never commit secrets** - Always use GitHub Secrets
2. **Rotate tokens regularly** - Every 90 days recommended
3. **Use least privilege** - Minimal token permissions
4. **Enable branch protection** - Require PR reviews for main branch
5. **Scan images** - Consider adding security scanning step

## Next Steps

1. Set up ArgoCD for automated GitOps deployment
2. Add automated testing before build
3. Implement staging environment
4. Add monitoring and alerting
5. Set up automated rollbacks
