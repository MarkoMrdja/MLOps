import torch
import torch.optim as optim

from data.load_data import load_data
from data.blob_storage import upload_model

from models.hyperoptimize import run_optimization
from models.train import train_model
from models.evaluate import evaluate_model
from models.fashion_cnn import FashionCNN
from utils.logger import logger


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = load_data()

best_params = run_optimization(train_loader, test_loader, device)

best_model = FashionCNN(
        num_filters_layer1=best_params['num_filters_layer1'],
        num_filters_layer2=best_params['num_filters_layer2'],
        kernel_size_layer1=best_params['kernel_size_layer1'],
        kernel_size_layer2=best_params['kernel_size_layer2'],
        fc1_units=best_params['fc1_units'],
        dropout_rate=best_params['dropout_rate'],
        activation_function=best_params['activation_function']
    ).to(device)

logger.info(f"Best model architecture: {best_model}")

optimizer = getattr(optim, best_params['optimizer'])(best_model.parameters(), lr=best_params['learning_rate'])

trained_model = train_model(model=best_model, train_loader=train_loader, optimizer=optimizer, device=device)

val_loss, val_accuracy = evaluate_model(model=trained_model, test_loader=test_loader, device=device)

upload_model(trained_model, val_accuracy)