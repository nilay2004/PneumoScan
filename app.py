from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
VISUALIZATIONS_FOLDER = 'visualizations/'
app.config['VISUALIZATIONS_FOLDER'] = VISUALIZATIONS_FOLDER
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model as None
model = None

def load_model():
    global model
    model_path = os.path.join('models', 'vgg16_model_new.keras')
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            return True
        else:
            logger.warning(f"Model file not found at {model_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/applications')
def applications():
    return render_template('applications.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        if not load_model():
            return jsonify({
                'error': 'Model not available. Please contact the administrator.',
                'status': 'error'
            }), 503

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            img = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(img)
            probability = float(prediction[0][0])
            
            result = {
                'prediction': 'Pneumonia' if probability > 0.5 else 'Normal',
                'probability': probability,
                'image_path': f'/static/uploads/{filename}'
            }
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Try to load the model at startup
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)