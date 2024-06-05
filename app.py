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
MODEL_PATH = 'D:\API\models\cnn-model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# print('Model loaded. Start serving...')


#function for processing the input image abd prediction
def model_predict(img_path, model):

    # Preprocessing the image
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    y = model.predict(x)

    return np.argmax(y)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        

        # Process your result for human
        dic= {0:"MildDemented", 1:"ModerateDemented" , 2:"VeryMildDemented" , 3:"NonDemented"}
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = dic[preds]   # ImageNet Decode
        # return the result
        return pred_class
    return None


if __name__ == '__main__':
    app.run(debug=True)