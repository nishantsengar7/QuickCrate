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
RUN chown -R user:user /var/log/nginx /var/lib/nginx

# Switch to non-root user
USER user

# Expose the expected Hugging Face Space port
EXPOSE 7860

# Start FastAPI (uvicorn) on 8001, Streamlit on 8501, and Nginx on 7860 in the foreground.
# Since Nginx runs in the foreground, it keeps the container alive.
CMD python -m uvicorn api:app --host 127.0.0.1 --port 8001 & \
    python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.enableCORS=false --server.enableXsrfProtection=false & \
    nginx -g "daemon off;"
