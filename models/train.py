import torch.nn as nn

def train_model(model, train_loader, optimizer, device, criterion=nn.CrossEntropyLoss(), num_epochs=1, return_avg_loss=False):
    """
    Train the model for a specified number of epochs and optionally return the average loss.

    Args:
        model (nn.Module): The neural network model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        train_loader (DataLoader): The DataLoader for training data.
        device (str): The device to run the training on ('cpu' or 'cuda').
        num_epochs (int): The number of epochs to train the model.
        return_avg_loss (bool): Whether to calculate and return the average loss.

    Returns:
        model (nn.Module): The trained model.
        avg_loss (float, optional): The average training loss, returned if return_avg_loss is True.
    """
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    if return_avg_loss:
        avg_loss = running_loss / len(train_loader)
        return model, avg_loss
    else:
        return model
