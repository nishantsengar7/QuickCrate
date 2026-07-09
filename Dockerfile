# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    QC_API_URL=http://localhost:8001

# Install system dependencies (nginx)
RUN apt-get update && apt-get install -y nginx && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 to match HF Spaces expectations
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and change ownership to user
COPY --chown=user:user . /app

# Copy Nginx config
COPY --chown=user:user nginx.conf /etc/nginx/nginx.conf

# Give the user permission to run nginx (write logs & run process)
RUN chmod +x /app/start_services.sh
RUN chown -R user:user /var/log/nginx /var/lib/nginx

# Switch to non-root user
USER user

# Pre-download and cache models to avoid 502 timeouts on cold starts
# BAAI/bge-large-en-v1.5 (~1.34 GB) and cross-encoder/ms-marco-MiniLM-L-6-v2 (~70 MB)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-large-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Expose the expected Hugging Face Space port
EXPOSE 7860

# Start services using the startup script
CMD ["/bin/bash", "/app/start_services.sh"]
