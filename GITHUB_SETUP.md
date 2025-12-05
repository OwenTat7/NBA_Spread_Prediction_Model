# GitHub Repository Setup Guide

## Step 1: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Repository settings:
   - **Name**: `NBA_Prediction_Model` (or your preferred name)
   - **Description**: "Machine learning model for predicting NBA game spread outcomes"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

### If you're creating a NEW repository on GitHub:

```bash
cd "/Users/owentatlonghari/Library/CloudStorage/OneDrive-UniversityofDelaware-o365/NBA_Prediction_Model"

# Add all files (respecting .gitignore)
git add .

# Commit your changes
git commit -m "Initial commit: NBA Spread Prediction Model"

# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/NBA_Prediction_Model.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### If you already have a remote (updating existing):

```bash
cd "/Users/owentatlonghari/Library/CloudStorage/OneDrive-UniversityofDelaware-o365/NBA_Prediction_Model"

# Check current remote
git remote -v

# If you need to update the remote URL:
# git remote set-url origin https://github.com/YOUR_USERNAME/NBA_Prediction_Model.git

# Add and commit changes
git add .
git commit -m "Update project: Add automation, organize structure, add Quarto report"

# Push to GitHub
git push origin main
```

## Step 3: Verify .gitignore is Working

The `.gitignore` file excludes:
- ✅ Data files (`data/*.csv`)
- ✅ Predictions (`predictions/`)
- ✅ Logs (`logs/`)
- ✅ Python cache (`__pycache__/`)
- ✅ Large files and temporary files

**Important**: These files will NOT be uploaded to GitHub (they're too large/private).

## Step 4: GitHub Pages (Optional - Host Your Report)

If you want to host your `index.html` report on GitHub Pages:

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select **"Deploy from a branch"**
4. Choose **main** branch and **/ (root)** folder
5. Click **Save**
6. Your report will be available at: `https://YOUR_USERNAME.github.io/NBA_Prediction_Model/`

**Note**: You'll need to render `index.qmd` to `index.html` first and commit it.

## Troubleshooting

### Authentication Issues

If you get authentication errors, you may need to:
- Use a Personal Access Token instead of password
- Set up SSH keys
- Use GitHub CLI: `gh auth login`

### Large Files

If you accidentally try to push large files:
```bash
# Remove from git cache
git rm --cached data/large_file.csv

# Update .gitignore if needed
# Then commit
git commit -m "Remove large files"
```

### Check What Will Be Pushed

```bash
# See what files are staged
git status

# See file sizes
git ls-files | xargs ls -lh | sort -k5 -hr | head -20
```

