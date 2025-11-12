#!/usr/bin/env python3
"""
Production entry point for SkillsMatch.AI on Render.com
"""
import os
import sys
from pathlib import Path

# Set production environment variables
os.environ['RENDER'] = '1'
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_DEBUG'] = 'False'

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'web'))

# Import Flask app for gunicorn
try:
    from web.app import app
    # Export app for gunicorn
    application = app
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

if __name__ == '__main__':
    # For direct running (development)
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False)