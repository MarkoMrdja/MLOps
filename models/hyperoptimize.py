import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

from models.fashion_cnn import FashionCNN, FashionCNN_Modular
from models.evaluate import evaluate_model
from models.train import train_model


# Run optimization
def run_optimization(train_loader, val_loader):
    """
    Runs the hyperparameter optimization using Hyperopt's TPE algorithm.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = 1
    max_evals = 2  #default: 10

    # Objective function
    def objective(params):
        """
        Objective function for hyperparameter optimization.
        Trains the model with the given hyperparameters and evaluates its performance.
        """

        # Model
        model = FashionCNN_Modular(
            num_filters_layer1=params['num_filters_layer1'],
            num_filters_layer2=params['num_filters_layer2'],
            kernel_size_layer1=params['kernel_size_layer1'],
            kernel_size_layer2=params['kernel_size_layer2'],
            fc1_units=params['fc1_units'],
            dropout_rate=params['dropout_rate'],
            activation_function=params['activation_function']
        ).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['lr'])

        # Training loop
        model = train_model(model, criterion, train_loader, optimizer, num_epochs, device)

        # Calculate accuracy
        val_loss, val_accuracy = evaluate_model(model, criterion, val_loader, device)

        # Return dictionary with the negative accuracy that it's trying to minimize and status
        return {'loss': -val_accuracy, 'status': STATUS_OK}

    # Define the search space
    search_space = {
        'num_filters_layer1': hp.choice('num_filters_layer1', [32, 64, 128]),
        'num_filters_layer2': hp.choice('num_filters_layer2', [64, 128, 256]),
        'kernel_size_layer1': hp.choice('kernel_size_layer1', [3, 5]),
        'kernel_size_layer2': hp.choice('kernel_size_layer2', [3, 5]),
        'fc1_units': hp.choice('fc1_units', [128, 256, 512]),
        'dropout_rate': hp.uniform('dropout_rate', 0.3, 0.7),
        'activation_function': hp.choice('activation_function', ['ReLU', 'LeakyReLU', 'ELU']),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
        'lr': hp.loguniform('lr', -5, -3),
        'optimizer': hp.choice('optimizer', ['Adam', 'SGD'])
    }

    best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals)
    
    best_params = {
            'num_filters_layer1': [32, 64, 128][best['num_filters_layer1']],
            'num_filters_layer2': [64, 128, 256][best['num_filters_layer2']],
            'kernel_size_layer1': [3, 5][best['kernel_size_layer1']],
            'kernel_size_layer2': [3, 5][best['kernel_size_layer2']],
            'fc1_units': [128, 256, 512][best['fc1_units']],
            'dropout_rate': best['dropout_rate'],
            'activation_function': ['ReLU', 'LeakyReLU', 'ELU'][best['activation_function']],
            'learning_rate': best['learning_rate']
        }

    # Create the best model with the correct parameters
    best_model = FashionCNN_Modular(
        num_filters_layer1=best_params['num_filters_layer1'],
        num_filters_layer2=best_params['num_filters_layer2'],
        kernel_size_layer1=best_params['kernel_size_layer1'],
        kernel_size_layer2=best_params['kernel_size_layer2'],
        fc1_units=best_params['fc1_units'],
        dropout_rate=best_params['dropout_rate'],
        activation_function=best_params['activation_function']
    ).to(device)

    return best_model


if __name__ == "__main__":
    best_hyperparameters = run_optimization()
    print(f"Best hyperparameters: {best_hyperparameters}")
