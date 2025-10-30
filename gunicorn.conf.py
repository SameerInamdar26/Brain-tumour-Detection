# Gunicorn configuration file

# Number of worker processes (keep low on free tier to save memory)
workers = 1

# Threads per worker (helps handle multiple requests without spawning more workers)
threads = 2

# Timeout in seconds (increase so model inference doesn’t get killed too quickly)
timeout = 120

# Bind is handled by Render automatically, so no need to set host/port here
 
