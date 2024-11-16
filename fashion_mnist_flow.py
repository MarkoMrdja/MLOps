from prefect import flow, task, serve 
from prefect.logging import get_run_logger
from prefect.server.schemas.schedules import CronSchedule

import torch
import torch.optim as optim
from typing import Tuple, Dict, Any

from data.load_data import load_data
from data.blob_storage import upload_model
from models.hyperoptimize import run_optimization
from models.train import train_model
from models.evaluate import evaluate_model
from models.fashion_cnn import FashionCNN


class ModelTrainingError(Exception):
    """Custom exception for model training errors"""
    pass

@task(
    name="setup_device",
    retries=2,
    retry_delay_seconds=30,
    tags=["setup"]
)
def setup_device() -> torch.device:
    logger = get_run_logger()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

@task(
    name="load_training_data",
    retries=3,
    retry_delay_seconds=60,
    tags=["data"]
)
def load_training_data() -> Tuple[Any, Any]:
    logger = get_run_logger()
    try:
        train_loader, test_loader = load_data()
        logger.info("Data loading successful")
        return train_loader, test_loader
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise ModelTrainingError(f"Failed to load training data: {str(e)}")

@task(
    name="optimize_hyperparameters",
    retries=2,
    retry_delay_seconds=300,  # 5 minutes
    tags=["optimization"]
)
def optimize_hyperparameters(train_loader: Any, test_loader: Any, device: torch.device) -> Dict[str, Any]:
    logger = get_run_logger()
    try:
        best_params = run_optimization(train_loader, test_loader, device)
        logger.info(f"Best hyperparameters found: {best_params}")
        return best_params
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        raise ModelTrainingError(f"Hyperparameter optimization failed: {str(e)}")

@task(
    name="initialize_model",
    retries=2,
    retry_delay_seconds=30,
    tags=["model"]
)
def initialize_model(best_params: Dict[str, Any], device: torch.device) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    logger = get_run_logger()
    try:
        model = FashionCNN(
            num_filters_layer1=best_params['num_filters_layer1'],
            num_filters_layer2=best_params['num_filters_layer2'],
            kernel_size_layer1=best_params['kernel_size_layer1'],
            kernel_size_layer2=best_params['kernel_size_layer2'],
            fc1_units=best_params['fc1_units'],
            dropout_rate=best_params['dropout_rate'],
            activation_function=best_params['activation_function']
        ).to(device)
        
        optimizer = getattr(optim, best_params['optimizer'])(
            model.parameters(), 
            lr=best_params['learning_rate']
        )
        
        logger.info(f"Model initialized with architecture:\n{model}")
        return model, optimizer
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise ModelTrainingError(f"Model initialization failed: {str(e)}")

@task(
    name="train",
    retries=1,
    retry_delay_seconds=600,  # 10 minutes
    tags=["training"]
)
def train(
    model: torch.nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> torch.nn.Module:
    logger = get_run_logger()
    try:
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device
        )
        logger.info("Model training completed successfully")
        return trained_model
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise ModelTrainingError(f"Model training failed: {str(e)}")

@task(
    name="evaluate",
    retries=2,
    retry_delay_seconds=30,
    tags=["evaluation"]
)
def evaluate(
    model: torch.nn.Module,
    test_loader: Any,
    device: torch.device
) -> Tuple[float, float]:
    logger = get_run_logger()
    try:
        val_loss, val_accuracy = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device
        )
        logger.info(f"Evaluation results - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        return val_loss, val_accuracy
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise ModelTrainingError(f"Model evaluation failed: {str(e)}")

@task(
    name="save_model",
    retries=3,
    retry_delay_seconds=60,
    tags=["storage"]
)
def save_model(model: torch.nn.Module, accuracy: float):
    logger = get_run_logger()
    try:
        upload_model(model, accuracy)
        logger.info(f"Model saved successfully with accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Model saving failed: {str(e)}")
        raise ModelTrainingError(f"Failed to save model: {str(e)}")

@flow(
    name="fashion_mnist_training",
    description="End-to-end training pipeline for Fashion MNIST model",
    version="1.0",
    retries=1
)
def training_pipeline() -> Tuple[float, float]:
    logger = get_run_logger()
    logger.info("Starting training pipeline")
    
    try:
        # Set up device
        device = setup_device()
        
        # Load data
        train_loader, test_loader = load_training_data()
        
        # Optimize hyperparameters
        best_params = optimize_hyperparameters(train_loader, test_loader, device)
        
        # Initialize model and optimizer
        model, optimizer = initialize_model(best_params, device)
        
        # Train model
        trained_model = train(model, train_loader, optimizer, device)
        
        # Evaluate model
        val_loss, val_accuracy = evaluate(trained_model, test_loader, device)
        
        # Save model
        save_model(trained_model, val_accuracy)
        
        logger.info("Pipeline completed successfully")
        return val_loss, val_accuracy
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    training_pipeline.serve(
        name="daily-fashion-mnist-training",
        work_pool_name="my-pool",
        schedule=(CronSchedule(cron="0 2 * * *", timezone="UTC")),  # Runs at 2 AM UTC daily
        description="Daily training of Fashion MNIST model",
        tags=["ml", "training", "fashion-mnist"]
    )