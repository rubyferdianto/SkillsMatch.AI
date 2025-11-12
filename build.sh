#!/bin/bash
# Render.com build script
set -e

echo "🚀 Starting SkillsMatch.AI build for Render.com..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p uploads/resumes
mkdir -p profiles

echo "✅ Build completed successfully!"
echo "🌐 Ready for deployment on Render.com"