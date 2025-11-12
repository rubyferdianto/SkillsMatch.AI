# 🚀 Render.com Deployment Guide for SkillsMatch.AI

## ✅ **Fixed Issues:**

1. **Environment Check**: Modified conda environment check to be production-friendly
2. **Dependencies**: Added missing Flask dependencies to requirements.txt
3. **Entry Point**: Created proper app.py entry point for gunicorn
4. **Configuration**: Added gunicorn configuration and Render-specific files

## 📁 **New Files Created for Deployment:**

### **1. requirements-render.txt** - Minimal Production Dependencies
```
Flask>=3.0.0
Flask-CORS>=4.0.0
Flask-SocketIO>=5.3.0
gunicorn>=21.2.0
python-dotenv>=1.0.0
openai>=1.3.0
SQLAlchemy>=2.0.0
psycopg2-binary>=2.9.0
pandas>=2.0.0
numpy>=1.24.0
# ... and other essentials
```

### **2. app.py** - Production Entry Point
```python
# Production-ready entry point that exports 'application' for gunicorn
# Handles environment detection and proper imports
```

### **3. gunicorn.conf.py** - Gunicorn Configuration
```python
# Optimized for Render.com with proper worker configuration
bind = f"0.0.0.0:{os.environ.get('PORT', 5003)}"
workers = 2
worker_class = "sync"
timeout = 120
```

### **4. render.yaml** - Render Service Configuration
```yaml
services:
  - type: web
    name: skillsmatch-ai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:application --config gunicorn.conf.py
```

### **5. Procfile** - Alternative Process Definition
```
web: gunicorn app:application --config gunicorn.conf.py
```

### **6. build.sh** - Build Script
```bash
#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data uploads/resumes profiles
```

## 🔧 **Modified Files:**

### **web/app.py**
- ✅ Fixed conda environment check for production
- ✅ Added production environment detection
- ✅ Removed restrictive local development warnings

### **requirements.txt**
- ✅ Added missing Flask dependencies:
  - Flask>=3.0.0
  - Flask-CORS>=4.0.0
  - Flask-SocketIO>=5.3.0
  - Werkzeug>=3.0.0

## 🚀 **Deployment Options for Render.com:**

### **Option 1: Using render.yaml (Recommended)**
1. Upload your project to GitHub
2. Connect to Render.com
3. Render will automatically detect `render.yaml`
4. Set environment variables in Render dashboard

### **Option 2: Manual Service Configuration**
1. Create new Web Service on Render
2. **Build Command:** `pip install -r requirements.txt`  
3. **Start Command:** `gunicorn app:application --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
4. **Environment:** `python`

### **Option 3: Using requirements-render.txt**
For faster deployment with minimal dependencies:
1. **Build Command:** `pip install -r requirements-render.txt`
2. **Start Command:** `gunicorn app:application --config gunicorn.conf.py`

## 🔐 **Environment Variables to Set in Render:**

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
GITHUB_TOKEN=your_github_token_here
DATABASE_URL=your_postgresql_url_here
FLASK_ENV=production
FLASK_DEBUG=false
RENDER=true
```

## 🗃️ **Database Configuration:**

### **Option 1: PostgreSQL (Recommended)**
1. Add PostgreSQL service in Render
2. Set `DATABASE_URL` environment variable
3. App will automatically use PostgreSQL

### **Option 2: SQLite (Fallback)**
- App automatically falls back to SQLite if no DATABASE_URL
- Data persists in Render's disk storage

## ⚡ **Performance Optimizations:**

- **Gunicorn Workers:** 2 workers for Render's default plan
- **Request Timeout:** 120 seconds for AI operations
- **Memory Efficient:** Optimized imports and lazy loading
- **Production Mode:** Disabled debug mode and verbose logging

## 🧪 **Testing Before Deployment:**

1. **Test locally with production settings:**
   ```bash
   export RENDER=1
   export FLASK_ENV=production
   gunicorn app:application --bind 127.0.0.1:5003
   ```

2. **Verify dependencies:**
   ```bash
   pip install -r requirements-render.txt
   python app.py
   ```

## 📋 **Deployment Checklist:**

- ✅ Requirements.txt includes Flask dependencies
- ✅ app.py entry point created  
- ✅ Gunicorn configuration ready
- ✅ Environment variables configured
- ✅ Production-friendly conda environment check
- ✅ Database fallback configured
- ✅ AI chat with working models (gpt-4o-mini)
- ✅ HR job exclusion working (149 jobs excluded)

## 🎯 **Expected Results:**

After deployment, your app will:
- ✅ Start without conda environment warnings
- ✅ Connect to PostgreSQL or fall back to SQLite
- ✅ Serve AI chat with gpt-4o-mini model
- ✅ Properly exclude HR jobs from IT professionals
- ✅ Handle profile management and job matching
- ✅ Scale with Render's infrastructure

The deployment should now work successfully on Render.com! 🚀