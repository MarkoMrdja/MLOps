import torch
import os
from data.blob_storage import upload_model, get_best_model

def save_model(model, accuracy):
    """
    Save the model to the blob storage. If the blob storage isn't available, save it locally.

    Args:
        model: PyTorch model to save
        accuracy: Accuracy of the model
    """

    try:
        upload_model(model, accuracy)
    except Exception as e:
        print(f"Error uploading model to blob storage: {e}")
        print("Attempting to save model locally...")

        model_filename = f"model_{accuracy:.4f}.pth"
        local_path = f"./models/failed_to_upload/{model_filename}"

        torch.save(model, local_path)

        print(f"Model saved locally as: {model_filename}")

def load_model():
    """
    Load the best model from the blob storage. If the blob storage isn't available, load the model saved locally.

    Returns:
        model: The best PyTorch model
    """

    try:
        model = get_best_model()
    except Exception as e:
        print(f"Error loading model from blob storage: {e}")
        print("Attempting to load model locally...")

        local_dir = "./models/best_model/"
        
        files = os.listdir(local_dir)
        
        if not files:
            raise FileNotFoundError(f"No files found in {local_dir}")
        
        files.sort()
        
        local_path = os.path.join(local_dir, files[0])
        
        model = torch.load(local_path)

        print(f"Model loaded locally from: {local_path}")

    return model