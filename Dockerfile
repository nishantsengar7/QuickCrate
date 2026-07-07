# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    QC_API_URL=http://localhost:8001

# Create user with UID 1000 to match HF Spaces expectations
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and change ownership to user
COPY --chown=user:user . /app

# Switch to non-root user
USER user

# Expose the expected Hugging Face Space port
EXPOSE 7860

# Start FastAPI (uvicorn) as a background process and Streamlit as the foreground process.
# Streamlit runs on 7860 (publicly exposed), and connects to FastAPI on port 8001 internally.
CMD python -m uvicorn api:app --host 127.0.0.1 --port 8001 & python -m streamlit run app.py --server.port 7860 --server.address 0.0.0.0
