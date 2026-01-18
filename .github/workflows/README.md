# CI/CD Workflow

Single GitHub Actions workflow for automated CI/CD of the CodeBuggy application.

## Workflow

### ci-cd.yaml - CI/CD Pipeline

**Triggers:**
- Push to `main` branch (when `app/**` changes)
- Manual dispatch

**Steps:**
1. Build multi-arch Docker image (amd64, arm64)
2. Push to Docker Hub with tags: `latest`, `{sha}`, `{date}-{sha}`
3. Update GitOps repository with new image tag
4. Create deployment summary

**Workflow:**
```
Push to main → Build Docker Image → Push to Docker Hub → Update GitOps Repo → Summary
```

## Setup

### Required Secrets

Add these secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

| Secret | Description | Example |
|--------|-------------|---------|
| `DOCKER_PASSWORD` | Docker Hub access token | `dckr_pat_xxx...` |
| `GITOPS_TOKEN` | GitHub token for GitOps repo | `ghp_xxx...` |

**Note:** GitOps repository (`dhung0811/codebuggy-k8s`) is configured in the workflow.

### Setup Steps

#### 1. Docker Hub Token
```bash
# Create token at: https://hub.docker.com/settings/security
# Add as DOCKER_PASSWORD secret
```

#### 2. GitOps Token
```bash
# Create token at: https://github.com/settings/tokens
# Permissions: repo (full)
# Add as GITOPS_TOKEN secret
```

### 3. GitOps Repository

```bash
# Set GITOPS_TOKEN secret
# The repository dhung0811/codebuggy-k8s is already configured
```

## GitOps Repository Structure

Your GitOps repository structure:

```
codebuggy-k8s (dev branch)
└── dev/
    └── deployment.yaml
```

The pipeline automatically updates the image tag in `dev/deployment.yaml`.

## Usage

### Automatic Deployment

```bash
# Make changes
git add .
git commit -m "feat: add new feature"
git push origin main

# Pipeline automatically:
# 1. Builds image
# 2. Pushes to Docker Hub
# 3. Updates GitOps repo
```

### Manual Deployment

```bash
# Go to: Actions → CI/CD Pipeline → Run workflow
```

## Monitoring

### Check Workflow Status

```bash
# View in GitHub UI
https://github.com/username/repo/actions

# Or use GitHub CLI
gh run list
gh run watch
```

## Troubleshooting

### Build Fails

**Check:**
1. Dockerfile syntax
2. Dependencies in requirements.txt

**Debug:**
```bash
# Test build locally
cd app
docker build -t test .
```

### Push Fails

**Check:**
1. DOCKER_PASSWORD secret is set
2. Docker Hub credentials are valid

**Debug:**
```bash
# Test push locally
docker login
docker push dhung04/codebuggy:test
```

### GitOps Update Fails

**Check:**
1. GITOPS_TOKEN has correct permissions
2. GITOPS_REPO format is correct (owner/repo)
3. GitOps repo exists and is accessible

## Customization

### Change Docker Registry

Edit workflow:
```yaml
env:
  DOCKER_REGISTRY: ghcr.io  # Change from docker.io
  DOCKER_USERNAME: ${{ github.repository_owner }}
```

### Add Security Scanning

```yaml
- name: Scan image
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: dhung04/codebuggy:${{ steps.vars.outputs.short-sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'
```

## Next Steps

1. Set up secrets in GitHub repository
2. Create GitOps repository with proper structure
3. Test pipeline with a small change
4. Configure ArgoCD/Flux to watch GitOps repo

---

**Last Updated**: 2026-01-18
