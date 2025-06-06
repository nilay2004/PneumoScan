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

# Load the new model
model_path = os.path.join('models', 'vgg16_model_new.keras')
try:
    model = tf.keras.models.load_model(model_path)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/')
def index():
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400

    try:
        # Create a secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(filepath)
        
        # Verify the file was saved
        if not os.path.exists(filepath):
            return jsonify({'error': 'Failed to save uploaded file'}), 500

        # Preprocess and predict
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
        
        # Return relative path for the image
        relative_path = os.path.join('uploads', filename)
        return jsonify({
            'result': result,
            'image_path': relative_path
        })

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Error processing image. Please try again.'}), 500


@app.route('/visualizations/<filename>')
def serve_visualization(filename):
    return send_from_directory(app.config['VISUALIZATIONS_FOLDER'], filename)


@app.route('/static/images/<filename>')
def serve_placeholder(filename):
    return send_from_directory('static/images', filename)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/submit-contact', methods=['POST'])
def submit_contact():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')
    captcha = request.form.get('captcha')

    if not all([name, email, subject, message]) or captcha != '8':
        return jsonify({'error': 'Invalid input or CAPTCHA. Please try again.'}), 400

    logging.info(f"Contact Form: {name}, {email}, {subject}, {message}")
    return jsonify({'message': 'Thank you for your message!'}), 200


if __name__ == '__main__':
    app.run(debug=True)