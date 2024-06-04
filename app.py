import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained SVM model
loaded_model = joblib.load('/home/khundmeer/Desktop/assignment/model.sav')  # Use the correct path to your model file

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """
    Render the main page where users can upload an image.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle the file upload and prediction.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('result.html', prediction=prediction)
        return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess_image(image_path):
    """
    Preprocess the input image for prediction.
    """
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (100, 100))  # Adjust size based on model training
        image = np.ravel(image).reshape(1, -1)  # Flatten the image to a 1D array
        return image
    except Exception as e:
        raise ValueError(f"Error in preprocessing image: {str(e)}")

def predict_image(image_path):
    """
    Predict the class of the input image using the loaded model.
    """
    try:
        processed_image = preprocess_image(image_path)
        y_pred = loaded_model.predict(processed_image)
        return str(y_pred[0])
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True)

