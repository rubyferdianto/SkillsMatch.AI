#!/bin/bash
# Debug runner with conda activation
# Usage: ./debug_run.sh

echo "🔧 Debug Runner - Ensuring smai environment is active"

# Source the activation helper
source ./activate_smai.sh

if [ $? -eq 0 ]; then
    echo "🌐 Starting Flask app in debug mode..."
    cd web
    python app.py
else
    echo "❌ Failed to activate environment"
    exit 1
fi