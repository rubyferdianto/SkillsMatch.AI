#!/bin/bash
# SkillsMatch.AI Environment Verification Script
# Usage: ./check_smai_env.sh

echo "🔍 SkillsMatch.AI Environment Verification"
echo "=========================================="

# Source conda activation
source ./activate_smai.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "📦 Environment Details:"
    echo "  - Conda Environment: $CONDA_DEFAULT_ENV"
    echo "  - Python Path: $(which python)"
    echo "  - Python Version: $(python --version)"
    echo "  - Working Directory: $(pwd)"
    echo ""
    
    # Test Python imports
    echo "🧪 Testing Key Imports:"
    python -c "
import sys
print(f'  ✅ Python: {sys.version.split()[0]}')

try:
    import flask
    print(f'  ✅ Flask: {flask.__version__}')
except ImportError as e:
    print(f'  ❌ Flask: Not installed')

try:
    import sqlalchemy
    print(f'  ✅ SQLAlchemy: {sqlalchemy.__version__}')
except ImportError:
    print(f'  ❌ SQLAlchemy: Not installed')

try:
    from database.db_config import db_config
    print('  ✅ Database Config: Available')
except ImportError as e:
    print(f'  ❌ Database Config: {e}')

try:
    from storage import profile_manager
    print('  ✅ Profile Manager: Available')
except ImportError as e:
    print(f'  ❌ Profile Manager: {e}')
"
    
    echo ""
    echo "🎯 Ready to run SkillsMatch.AI!"
    echo "💡 Use: ./start_skillmatch.sh to start the application"
    
else
    echo "❌ Environment activation failed!"
    exit 1
fi