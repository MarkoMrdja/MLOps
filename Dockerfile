FROM python:3.11-slim

# Install system dependencies including supervisor and curl (for healthcheck)
RUN apt-get update && apt-get install -y \
    gcc \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.linux.txt .
RUN pip install -r requirements.linux.txt

# Create supervisor directories
RUN mkdir -p /var/log/supervisor /var/run/supervisor /etc/supervisor/conf.d

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy application code
COPY . .