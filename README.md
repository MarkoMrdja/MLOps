# MLOps Project

A containerized machine learning pipeline for Fashion MNIST image classification with automated training, hyperparameter optimization, and prediction serving.

## Overview

This project implements a complete MLOps workflow for training, optimizing, and deploying a CNN model that classifies fashion items from the Fashion MNIST dataset. The workflow is orchestrated using Prefect, containerized with Docker, and exposes predictions through a Flask REST API.

## Architecture

- **Model**: Convolutional Neural Network with configurable architecture
- **Data Pipeline**: Preprocessing, augmentation, and Azure Blob Storage integration
- **Training Pipeline**: Prefect-orchestrated workflow for regular retraining
- **Hyperparameter Optimization**: Automated tuning using Hyperopt
- **Serving**: Flask API with rate limiting and model reloading
- **Deployment**: Docker containerization with supervisord process management

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Azure Storage account (for model persistence)

### Installation

1. Clone the repository
```bash
git clone https://github.com/MarkoMrdja/MLOps.git
cd MLOps
```

2. Create .env file and configure Azure environment variables
```bash
AZURE_CLIENT_ID=<client-id>
AZURE_CLIENT_SECRET=<client-secret>
AZURE_TENANT_ID=<tenant-id>
AZURE_STORAGE_URL=https://<your-storage-account>.blob.core.windows.net/
```

### Deployment

```bash
docker-compose up -d
```

All services are automatically managed by supervisord inside the container:
- Prefect Server
- Prefect Worker
- Model Training Pipeline
- Flask API Server

## API Usage

The model is served via REST API at `http://localhost:5000/predict`

### Sending an Image for Prediction
You can send an image file to be classified using a standard multipart/form-data POST request:

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/your/image.jpg"
```

Response:
```json
{
  "predicted_class": "Ankle boot"
}
```

## Workflow Orchestration

The MLOps pipeline is orchestrated with Prefect:

1. Data loading and preprocessing
2. Hyperparameter optimization
3. Model training with best parameters
4. Model evaluation and versioning
5. Deployment of best model

## Project Structure

- `/api` - Flask API implementation
- `/data` - Data loading, preprocessing, and storage utilities
- `/models` - CNN model definition, training, and evaluation code
- `/utils` - Helper functions and utilities
- `fashion_mnist_flow.py` - Main Prefect workflow definition
- `app.py` - API server entrypoint
- `supervisord.conf` - Process management configuration

## Academic Project Information

This project was developed as part of the MLOps course for my Artificial Intelligence & Machine Learning Master's program. It demonstrates the practical application of MLOps principles and tools in building and deploying a complete machine learning pipeline with industry-standard practices.
