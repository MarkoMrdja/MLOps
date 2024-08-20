import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_data(batch_size=64):
    """
    Load the Fashion MNIST dataset using defined transformations.
    
    Args:
        batch_size (int): The batch size for the data loaders.
        
    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    # Get the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load the datasets with transformations
    train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
