from flask import Flask, request, jsonify
import torch
from PIL import Image
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils.save_load import load_model
from data.preprocess import get_request_data_transformations
from utils.label_mapping import output_label
from utils.logger import logger

app = Flask(__name__)

# Rate limiter setup
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Load the model in memory
logger.info("Loading initial model...")
model = load_model()
model.eval()
logger.info("Model loaded and set to evaluation mode")


@app.route('/predict', methods=['POST'])
def predict():
    client_ip = request.remote_addr
    logger.info(f"Prediction request from {client_ip}")
    
    try:
        if 'file' not in request.files:
            logger.warning(f"No file in request from {client_ip}")
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.warning(f"Empty filename from {client_ip}")
            return jsonify({'error': 'No file selected'}), 400
            
        logger.debug(f"Processing file: {file.filename}")
        
        img = Image.open(file).convert('RGB')
        transform = get_request_data_transformations()
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        
        output_lbl = output_label(predicted)
        
        logger.info(f"Successful prediction for {file.filename}: {output_lbl}")
        return jsonify({'predicted_class': output_lbl})
        
    except Exception as e:
        logger.error(f"Prediction error for {client_ip}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/reload_model', methods=['POST'])
@limiter.limit("1 per 5 minutes")
def reload_model():
    logger.info("Model reload requested")
    try:
        global model
        model = load_model()
        model.eval()
        logger.info("Model reloaded successfully")
        return jsonify({'message': 'Model reloaded successfully'})
    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")
        return jsonify({'error': 'Model reload failed'}), 500