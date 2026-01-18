# GitHub Actions CI/CD

Single workflow for automated CI/CD pipeline.

## Workflow

### CI/CD Pipeline (`ci-cd.yaml`)
Builds, pushes Docker images, and updates GitOps repository.

**Triggers:**
- Push to `main` branch (when `app/**` changes)
- Manual dispatch

**Steps:**
1. Build multi-arch Docker image (amd64, arm64)
2. Push to Docker Hub with tags: `latest`, `{sha}`, `{date}-{sha}`
3. Update GitOps repository with new image tag
4. Create deployment summary

## Setup

### Required Secrets

Configure in GitHub repository settings (Settings → Secrets and variables → Actions):

- `DOCKER_PASSWORD`: Docker Hub access token
- `GITOPS_REPO`: GitOps repository (format: `owner/repo`)
- `GITOPS_TOKEN`: GitHub token with repo access

### Quick Start

1. Set up the required secrets in GitHub
2. Push changes to `main` branch or trigger manually
3. Pipeline will automatically build, push, and update GitOps repo

## GitOps Integration

The pipeline automatically updates your GitOps repository after successful builds. See [GITOPS_TEMPLATE.md](./GITOPS_TEMPLATE.md) for GitOps repository structure.

## Documentation

- [CI_CD_SETUP_GUIDE.md](./CI_CD_SETUP_GUIDE.md) - Complete setup instructions
- [GITOPS_TEMPLATE.md](./GITOPS_TEMPLATE.md) - GitOps repository template
