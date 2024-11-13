# Stage 1: Builder stage
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim as runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY ./api ./api
COPY ./data ./data
COPY ./models ./models
COPY ./pipelines ./pipelines
COPY ./utils ./utils
COPY app.py .
COPY .env .

# Create necessary directories
RUN mkdir -p models/best_model models/failed_to_upload

# MetaFlow configuration
ENV METAFLOW_DATASTORE_SYSROOT_LOCAL=/metaflow
ENV METAFLOW_DEFAULT_DATASTORE=local
ENV METAFLOW_DEFAULT_METADATA=local

# Create MetaFlow directory
RUN mkdir -p /metaflow

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]