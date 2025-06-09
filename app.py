from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the SavedModel
model = tf.keras.models.load_model('models/rice_leaf_disease_model.h5')

# Class names
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast', 'Leaf scald', 'Sheath Blight']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_image(image):
    img_array = np.array(image)
    if np.all(img_array == 0):
        return False, "Image is completely black"
    if np.all(img_array == 255):
        return False, "Image is completely white"
    if np.std(img_array) < 10:
        return False, "Image has very low contrast"
    if np.mean(img_array) < 30:
        return False, "Image is too dark"
    return True, "Image is valid"

def preprocess_image(image):
    is_valid, message = is_valid_image(image)
    if not is_valid:
        raise ValueError(message)
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG'}), 400
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > MAX_FILE_SIZE:
            return jsonify({'success': False, 'error': 'File too large. Maximum size is 5MB'}), 400
        image = Image.open(file.stream).convert('RGB')
        is_valid, message = is_valid_image(image)
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence
        })
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        test_input = np.random.random((1, 128, 128, 3))
        model.predict(test_input)
        return jsonify({
            'status': 'healthy',
            'model_loaded': True
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)