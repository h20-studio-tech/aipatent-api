#!/bin/bash
set -e

# Print deployment start message
echo "🚀 Starting deployment for AI Patent API"

# Fetch latest changes
echo "📥 Pulling latest changes from main branch"
git fetch --all
git reset --hard origin/main

# Install uv if not exists
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Sync dependencies with uv (handles virtual environment automatically)
echo "📦 Installing dependencies with uv"
uv sync

# Restart the service
echo "🔄 Restarting the API service"
sudo systemctl restart aipatent-api.service

# Check the service status
echo "✅ Checking service status"
sudo systemctl status aipatent-api.service --no-pager

echo "🎉 Deployment completed!"