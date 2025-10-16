# GitHub Actions Deployment Setup

This guide explains how to set up automated deployment to Scaleway for the Halo backend.

## Required GitHub Secrets

You need to add the following secrets to your GitHub repository:

### 1. Get Scaleway Credentials

Run these commands locally to get your Scaleway credentials:

```bash
# Access Key
scw config get access-key

# Secret Key
scw config get secret-key

# Organization ID
scw config get default-organization-id
```

### 2. Create Scaleway Registry Token

1. Go to [Scaleway Console](https://console.scaleway.com/)
2. Navigate to: **Container Registry** → **funcscwhalobackend4k1irws6**
3. Click **Credentials** → **Generate credentials**
4. Copy the generated token (it will only be shown once!)

### 3. Add Secrets to GitHub

1. Go to your GitHub repository: `https://github.com/ArcheronTechnologies/atlas-halo-backend`
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add each of these:

| Secret Name | Value | Where to Get It |
|-------------|-------|-----------------|
| `SCW_ACCESS_KEY` | Your access key | `scw config get access-key` |
| `SCW_SECRET_KEY` | Your secret key | `scw config get secret-key` |
| `SCW_DEFAULT_ORGANIZATION_ID` | Your org ID | `scw config get default-organization-id` |
| `SCW_REGISTRY_TOKEN` | Registry token | Scaleway Console → Container Registry → Credentials |

## How It Works

The workflow (`.github/workflows/scaleway-deploy.yml`) automatically:

1. **Triggers** on push to `main` branch when backend files change
2. **Builds** AMD64 Docker image (compatible with Scaleway)
3. **Pushes** to Scaleway Container Registry
4. **Updates** the container to use the new image
5. **Verifies** deployment by testing health endpoints

## Manual Deployment Trigger

If you want to manually trigger deployment after setting up secrets:

```bash
# Make a small change and push
cd /path/to/Halo
git add .github/workflows/scaleway-deploy.yml
git commit -m "Add Scaleway deployment workflow"
git push backend-origin main
```

## Monitoring Deployments

1. Go to: **Actions** tab in GitHub repository
2. Click on the latest **Deploy Backend to Scaleway** workflow run
3. Watch real-time logs for each deployment step

## Deployment Stages

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Code Push to main                                        │
│    Files: backend/*, main.py, requirements.txt, Dockerfile  │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GitHub Actions Runner (Ubuntu AMD64)                     │
│    - Checkout code                                          │
│    - Build Docker image with --platform=linux/amd64         │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Push to Scaleway Registry                                │
│    rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/          │
│      halo-backend:latest                                    │
│      halo-backend:<git-sha>                                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Update Serverless Container                              │
│    - Container ID: 35a73370-0199-42de-862c-88b67af1890d     │
│    - Pull new image                                         │
│    - Redeploy with zero downtime                            │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Verify Deployment                                        │
│    - Test /health endpoint                                  │
│    - Test /api/v1/ai/health endpoint                        │
│    - ✅ Success or ❌ Rollback                               │
└─────────────────────────────────────────────────────────────┘
```

## Rollback Procedure

If a deployment fails, rollback to previous version:

```bash
# List available tags
scw registry tag list image-id=69be41ca-f518-4fab-b72d-80fd6e02c6dd

# Update to previous working tag
scw container container update 35a73370-0199-42de-862c-88b67af1890d \
  registry-image="rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:categories-fix" \
  redeploy=true --wait
```

## Current Deployment Status

**Before Setup**: Manual deployment only, sensor fusion code not in production
**After Setup**: Automatic deployment on every push to main, sensor fusion live

## Testing the Workflow

After setting up secrets, test the workflow:

```bash
# 1. Make a trivial change
echo "# Deployment test" >> backend/README.md

# 2. Commit and push
git add backend/README.md
git commit -m "Test deployment workflow"
git push backend-origin main

# 3. Watch deployment
# Go to GitHub Actions tab and monitor the workflow

# 4. Verify deployment
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health
```

## Troubleshooting

### Error: "Authentication failed"
- Check that `SCW_REGISTRY_TOKEN` is correctly set in GitHub secrets
- Token may have expired - regenerate in Scaleway Console

### Error: "Invalid Image architecture"
- Ensure workflow is building with `--platform=linux/amd64`
- Check that GitHub Actions runner is using Ubuntu (not macOS)

### Error: "Container update failed"
- Verify container ID is correct: `35a73370-0199-42de-862c-88b67af1890d`
- Check Scaleway Console for container errors

### Deployment Stuck
- Check container logs: `scw container container logs 35a73370-0199-42de-862c-88b67af1890d`
- Verify health endpoints are responding

## Security Notes

- Secrets are encrypted by GitHub and never logged
- Registry token has push-only access to the specific namespace
- Workflow runs in isolated GitHub-hosted runner
- Each deployment creates immutable image tagged with git SHA
