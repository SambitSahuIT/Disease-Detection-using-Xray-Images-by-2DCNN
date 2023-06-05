from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import sys

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Detection_Covid_19.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
#model._make_predict_function() 
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')

print('Model loaded. Check http://127.0.0.1:8000/')


def model_predict(img_path, model):
    xtest_image = image.load_img(img_path, target_size=(224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis = 0)
    preds = (model.predict(xtest_image) > 0.5).astype("int32")
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
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
        
        if preds[0][0] == 0:
            prediction = 'Positive For Covid-19'
        else:
            prediction = 'Negative for Covid-19'
        return prediction
    return None

if __name__ == '__main__':
    # Start the Flask app with Waitress
    from waitress import serve
    serve(app, host='0.0.0.0', port=8000, threads=1, channel_timeout=300)
