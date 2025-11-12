# Gunicorn configuration for Render.com deployment
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5003)}"
backlog = 2048

# Worker processes
workers = 2
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 120
keepalive = 2

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "skillsmatch-ai"

# Server mechanics
daemon = False
pidfile = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None