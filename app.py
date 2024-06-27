from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model ,load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename






# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'cnn-model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# print('Model loaded. Start serving...')

preds = ['MildDemented' ,'ModerateDemented', 'NonDemented', 'VeryMildDemented']

app.config['UPLOAD_FOLDER'] = 'uploads'
#function for processing the input image abd prediction
def model_predict(img_path, model):

    # Preprocessing the image
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    y = model.predict(x)

    return preds[np.argmax(y)]



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    f = request.files['file']
    if f.filename == '':
        return "No selected file"
    if f:
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        return preds  # Make sure to return the prediction result

if __name__ == '__main__':
    app.run(debug=True)