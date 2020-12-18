from flask import Flask, redirect, url_for, render_template, request
from keras.models import load_model
import cv2             
import numpy as np  
from tqdm import tqdm
import os                   
from werkzeug.utils import secure_filename
# import tensorflow as tf

# export_dir='model.h5'
# converter = tf.lite.TFLiteConverter.from_keras_model(export_dir)
# model = converter.convert()

model = load_model('./model.h5')

app = Flask(__name__,template_folder='template')

@app.route("/upload-image", methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        f = request.files['image']
        f_name = 'images/' + secure_filename(f.filename)
        f.save(f_name)
        fileread=cv2.imread(f_name)
        fileread=cv2.resize(fileread,(150,150))
        fileread = np.array(fileread)
        fileread = np.expand_dims(fileread, axis=0)
        pred = np.round(model.predict(fileread))
        if pred[0][0] == 0 and pred[0][1] == 1:
            return render_template('NoFireDetected.html')
        else:
            return render_template('FireDetected.html')

    return render_template("upload_image.html")

if __name__ == "__main__":
    app.run(debug=True)
