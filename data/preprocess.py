import torchvision.transforms as transforms

def get_dataset_transformations():
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

def get_request_data_transformations():
    """
    Define and return the transformations for the request data.
    
    Returns:
        transform (transforms.Compose): Composed transformations.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    return transform