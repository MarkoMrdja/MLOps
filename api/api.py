from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils.save_load import load_model
from data.preprocess import get_request_data_transformations
from utils.label_mapping import output_label

app = Flask(__name__)

# Rate limiter setup
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Load the model in memory
model = load_model()
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    transform = get_request_data_transformations()
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    
    output_lbl = output_label(predicted)

    return jsonify({'predicted_class': output_lbl})


@app.route('/reload_model', methods=['POST'])
@limiter.limit("1 per 5 minutes")
def reload_model():
    global model
    model = load_model()
    model.eval()
    return jsonify({'message': 'Model reloaded successfully'})