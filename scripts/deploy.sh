#!/bin/bash
set -e

# Print deployment start message
echo "🚀 Starting deployment for AI Patent API"

# Fetch latest changes
echo "📥 Pulling latest changes from main branch"
git fetch --all
git reset --hard origin/main

echo "📦 Activating virtual environment"
source .venv/bin/activate

echo "📦 Installing dependencies with pip"
pip install -r requirements.txt

# Restart the service
echo "🔄 Restarting the API service"
sudo systemctl restart aipatent-api.service

# Check the service status
echo "✅ Checking service status"
sudo systemctl status aipatent-api.service

echo "🎉 Deployment completed!"
