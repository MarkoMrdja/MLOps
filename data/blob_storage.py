from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
import torch
import os
import io

load_dotenv()

client_id = os.environ["AZURE_CLIENT_ID"]
client_secret = os.environ["AZURE_CLIENT_SECRET"]
tenant_id = os.environ["AZURE_TENANT_ID"]
account_url = os.environ["AZURE_STORAGE_URL"]

credentials = ClientSecretCredential(
    client_id=client_id,
    client_secret=client_secret,
    tenant_id=tenant_id
)

container_name = "fashion-mnist-models"
blob_service_client = BlobServiceClient(account_url=account_url, credential=credentials)
container_client = blob_service_client.get_container_client(container_name)


def upload_model(model, accuracy):
    # Serialize the model to a byte stream
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)  # Rewind the buffer to the beginning

    # Define model filename with accuracy and current datetime
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    model_filename = f"model_{current_time}_{accuracy:.4f}.pth"

    # Upload to all_models directory
    all_models_path = f"all_models/{model_filename}"
    all_blob_client = container_client.get_blob_client(all_models_path)
    all_blob_client.upload_blob(buffer, overwrite=True)
    print(f"Uploaded to all_models: {model_filename}")

    # Check if this model is better than the current best
    best_model_filename = get_best_model_filename()
    if best_model_filename:
        current_best_accuracy = float(best_model_filename.split('_')[-1].replace('.pth', ''))
    else:
        current_best_accuracy = -1  # No best model exists yet

    if accuracy > current_best_accuracy:
        # Rewind the buffer again for uploading the best model
        buffer.seek(0)

        # Upload to best_model directory
        best_models_path = f"best_model/{model_filename}"
        best_blob_client = container_client.get_blob_client(best_models_path)
        best_blob_client.upload_blob(buffer, overwrite=True)
        print(f"New best model saved: {model_filename}")
        
        # Save locally as the best model
        local_best_model_path = f"./models/best_model/{model_filename}"
        os.makedirs(os.path.dirname(local_best_model_path), exist_ok=True)
        with open(local_best_model_path, "wb") as local_file:
            local_file.write(buffer.getvalue())
        print(f"Saved locally: {local_best_model_path}")

        # Delete the old best model from Azure
        if best_model_filename:
            container_client.delete_blob(best_model_filename)
            print(f"Deleted old best model: {best_model_filename}")
    else:
        print(f"Model {model_filename} is not better than the current best model.")

    buffer.close()


def get_best_model():
    best_model_filename = get_best_model_filename()
    if best_model_filename:
        blob_client = container_client.get_blob_client(best_model_filename)
        model_stream = blob_client.download_blob().readall()
        model = torch.load(io.BytesIO(model_stream))
        return model
    else:
        return None


def get_best_model_filename():
    blob_list = container_client.list_blobs(name_starts_with="best_model/")
    for blob in blob_list:
        return blob.name
    return None


def cleanup_old_models():
    # Remove models older than 30 days from all_models
    expiration_date = datetime.now() - timedelta(days=30)
    blob_list = container_client.list_blobs(name_starts_with="all_models/")
    
    for blob in blob_list:
        if blob.last_modified < expiration_date:
            container_client.delete_blob(blob)
            print(f"Deleted old model: {blob.name}")