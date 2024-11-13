from metaflow import FlowSpec, step, Parameter, current
import torch
import torch.optim as optim

from data.load_data import load_data
from data.blob_storage import upload_model
from models.hyperoptimize import run_optimization
from models.train import train_model
from models.evaluate import evaluate_model
from models.fashion_cnn import FashionCNN


class FashionMNISTFlow(FlowSpec):
    """
    A Metaflow pipeline for training a Fashion MNIST classifier.
    """

    @step
    def start(self):
        """
        Initialize pipeline and set up device.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Load and prepare the data.
        """
        self.train_loader, self.test_loader = load_data()
        print("Data loaded successfully")
        self.next(self.optimize_hyperparameters)

    @step
    def optimize_hyperparameters(self):
        """
        Run hyperparameter optimization.
        """
        self.best_params = run_optimization(
            train_loader=self.train_loader,
            val_loader=self.test_loader,
            device=self.device
        )
        print("Best parameters found:", self.best_params)
        self.next(self.create_model)

    @step
    def create_model(self):
        """
        Create the model with best parameters.
        """
        self.model = FashionCNN(
            num_filters_layer1=self.best_params['num_filters_layer1'],
            num_filters_layer2=self.best_params['num_filters_layer2'],
            kernel_size_layer1=self.best_params['kernel_size_layer1'],
            kernel_size_layer2=self.best_params['kernel_size_layer2'],
            fc1_units=self.best_params['fc1_units'],
            dropout_rate=self.best_params['dropout_rate'],
            activation_function=self.best_params['activation_function']
        ).to(self.device)

        self.optimizer = getattr(optim, self.best_params['optimizer'])(
            self.model.parameters(), 
            lr=self.best_params['learning_rate']
        )
        
        print("Model created:", self.model)
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train the model with the best hyperparameters.
        """
        self.trained_model = train_model(
            model=self.model,
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device
        )
        print("Model training completed")
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """
        Evaluate the trained model.
        """
        self.val_loss, self.val_accuracy = evaluate_model(
            model=self.trained_model,
            test_loader=self.test_loader,
            device=self.device
        )
        print(f"Validation Loss: {self.val_loss:.4f}")
        print(f"Validation Accuracy: {self.val_accuracy:.4f}")
        self.next(self.save_model)

    @step
    def save_model(self):
        """
        Save the trained model.
        """
        upload_model(self.trained_model, self.val_accuracy)
        print("Model saved successfully")
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        print("Pipeline completed successfully!")
        print(f"Final model accuracy: {self.val_accuracy:.4f}")
