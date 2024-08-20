import torch

def evaluate_model(model, criterion, test_loader, device):
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

    print(f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy * 100:.2f}%')

    return val_loss, val_accuracy
