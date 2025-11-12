#!/bin/bash
# Simple conda environment activation helper
# Usage: source ./activate_smai.sh  OR  . ./activate_smai.sh

echo "🔧 Activating smai conda environment..."

# Initialize conda in this shell (try multiple locations)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    echo "🐍 Using miniconda3"
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    echo "🐍 Using anaconda3"
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    echo "🐍 Using system conda"
else
    echo "❌ Error: Could not find conda installation!"
    return 1
fi

# Force conda activation
eval "$(conda shell.bash hook)"
conda activate smai

# Verify activation
if [ "$CONDA_DEFAULT_ENV" = "smai" ]; then
    echo "✅ smai environment activated successfully!"
    echo "🐍 Using Python: $(which python)"
    echo "📦 Active environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  Activation failed. Current environment: $CONDA_DEFAULT_ENV"
    echo "💡 Try: conda activate smai"
    return 1
fi