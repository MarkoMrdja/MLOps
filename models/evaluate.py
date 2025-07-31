import torch
import torch.nn as nn
from utils.logger import logger

def evaluate_model(model, test_loader, device, criterion=nn.CrossEntropyLoss()):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The trained neural network model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): The device to run the evaluation on ('cpu' or 'cuda').
        criterion (nn.Module): The loss function.

    Returns:
        val_loss (float): The average loss on the test set.
        val_accuracy (float): The accuracy on the test set.
    """

    running_loss = 0.0
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        val_loss = running_loss / len(test_loader)
        val_accuracy = correct / total

    logger.info(f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy * 100:.2f}%')

    return val_loss, val_accuracy
