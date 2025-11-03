from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load trained models for ASL and Gesture
asl_model = load_model("asl_model.h5")
gesture_model = load_model("gesture_model.h5")

# Load class labels for ASL and Gesture
asl_classes = np.load("label_classes.npy")  # Ensure correct path
gesture_classes = np.load("gesturelabel_classes.npy")  # Ensure correct path

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image before prediction
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 64, 64, 1))  # Adding batch dimension
    return reshaped

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    # Get model type (ASL or Gesture) from form
    model_type = request.form['model_type']

    # Check if the post request has the file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # If the file is allowed, process it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Read the image
        image = cv2.imread(filepath)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction based on the selected model
        if model_type == 'asl':
            prediction = asl_model.predict(processed_image)
            label = asl_classes[np.argmax(prediction)]
        elif model_type == 'gesture':
            prediction = gesture_model.predict(processed_image)
            label = gesture_classes[np.argmax(prediction)]
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        # Return prediction as JSON response
        return jsonify({'prediction': label}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
