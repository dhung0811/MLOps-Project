# GitOps Repository Template

## 📁 Recommended Structure

```
gitops-repo/
├── README.md
├── environments/
│   ├── staging/
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── pvc.yaml
│   └── production/
│       ├── kustomization.yaml
│       ├── namespace.yaml
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── configmap.yaml
│       └── pvc.yaml
└── base/
    ├── kustomization.yaml
    ├── deployment.yaml
    ├── service.yaml
    └── configmap.yaml
```

---

## 📝 File Templates

### environments/staging/kustomization.yaml

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: codebuggy-staging

resources:
  - namespace.yaml
  - deployment.yaml
  - service.yaml
  - configmap.yaml
  - pvc.yaml

images:
  - name: dhung04/codebuggy
    newTag: latest  # Updated by CI/CD

commonLabels:
  environment: staging
  app: codebuggy

configMapGenerator:
  - name: codebuggy-config
    literals:
      - MLFLOW_URI=http://mlflow.mlops.svc.cluster.local:5000
      - MODEL_STAGE=Staging
```

---

### environments/staging/namespace.yaml

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: codebuggy-staging
  labels:
    environment: staging
```

---

### environments/staging/deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codebuggy-app
  namespace: codebuggy-staging
spec:
  replicas: 1
  selector:
    matchLabels:
      app: codebuggy
  template:
    metadata:
      labels:
        app: codebuggy
        environment: staging
    spec:
      containers:
      - name: codebuggy
        image: dhung04/codebuggy:latest  # Updated by CI/CD
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
        env:
        - name: MLFLOW_URI
          value: "http://mlflow.mlops.svc.cluster.local:5000"
        - name: MODEL_STAGE
          value: "Staging"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

---

### environments/production/kustomization.yaml

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: codebuggy

resources:
  - namespace.yaml
  - deployment.yaml
  - service.yaml
  - configmap.yaml
  - pvc.yaml

images:
  - name: dhung04/codebuggy
    newTag: latest  # Updated by CI/CD

commonLabels:
  environment: production
  app: codebuggy

replicas:
  - name: codebuggy-app
    count: 3  # More replicas in production

configMapGenerator:
  - name: codebuggy-config
    literals:
      - MLFLOW_URI=http://mlflow.mlops.svc.cluster.local:5000
      - MODEL_STAGE=Production
```

---

### environments/production/deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codebuggy-app
  namespace: codebuggy
spec:
  replicas: 3  # More replicas in production
  selector:
    matchLabels:
      app: codebuggy
  template:
    metadata:
      labels:
        app: codebuggy
        environment: production
    spec:
      containers:
      - name: codebuggy
        image: dhung04/codebuggy:latest  # Updated by CI/CD
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
        env:
        - name: MLFLOW_URI
          value: "http://mlflow.mlops.svc.cluster.local:5000"
        - name: MODEL_STAGE
          value: "Production"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
```

---

## 🚀 Setup Instructions

### 1. Create GitOps Repository

```bash
# Create new repository on GitHub
gh repo create gitops-codebuggy --public

# Clone it
git clone https://github.com/username/gitops-codebuggy
cd gitops-codebuggy
```

### 2. Create Directory Structure

```bash
mkdir -p environments/{staging,production}
mkdir -p base

# Copy templates
# ... (copy files from above)
```

### 3. Initialize Git

```bash
git add .
git commit -m "Initial GitOps setup"
git push origin main
```

### 4. Configure ArgoCD

```yaml
# argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: codebuggy-staging
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/username/gitops-codebuggy
    targetRevision: main
    path: environments/staging
  destination:
    server: https://kubernetes.default.svc
    namespace: codebuggy-staging
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

Apply:
```bash
kubectl apply -f argocd-app.yaml
```

---

## 🔄 CI/CD Integration

The CI/CD pipeline will automatically update image tags in this repository:

```bash
# Example: After building image with tag abc123
# CI/CD updates:
environments/staging/kustomization.yaml:
  images:
    - name: dhung04/codebuggy
      newTag: abc123  # ← Updated

# ArgoCD detects change and syncs to cluster
```

---

## 📊 Monitoring

### ArgoCD UI

```bash
# Access ArgoCD
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Login
argocd login localhost:8080

# Check app status
argocd app get codebuggy-staging
```

### Sync Status

```bash
# Manual sync
argocd app sync codebuggy-staging

# Check diff
argocd app diff codebuggy-staging
```

---

## 🔐 Security

### Branch Protection

Enable in GitHub:
- Require pull request reviews
- Require status checks
- Restrict who can push

### RBAC

```yaml
# argocd-rbac.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-rbac-cm
  namespace: argocd
data:
  policy.csv: |
    p, role:developer, applications, get, */*, allow
    p, role:developer, applications, sync, */staging/*, allow
    p, role:admin, applications, *, */*, allow
```

---

## 📝 Best Practices

### 1. Separate Environments

✅ Different namespaces for staging/production  
✅ Different resource limits  
✅ Different replica counts

### 2. Use Kustomize

✅ Base configuration + overlays  
✅ Easy to manage multiple environments  
✅ No templating complexity

### 3. Automated Sync

✅ Enable auto-sync in ArgoCD  
✅ Enable self-heal  
✅ Enable prune

### 4. Version Control

✅ Tag releases  
✅ Use semantic versioning  
✅ Keep changelog

---

## 🎯 Example Workflow

```
Developer pushes code
    ↓
GitHub Actions builds image (tag: abc123)
    ↓
CI/CD updates GitOps repo
    ↓
GitOps repo commit: "Update image to abc123"
    ↓
ArgoCD detects change
    ↓
ArgoCD syncs to Kubernetes
    ↓
New pods deployed with image abc123
```

---

**Template Version**: 1.0  
**Last Updated**: 2026-01-18
