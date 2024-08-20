import torchvision.transforms as transforms

def get_transformations():
    """
    Define and return the transformations for the dataset.
    
    Returns:
        transform (transforms.Compose): Composed transformations.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    return transform
