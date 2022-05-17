from flask import Flask, flash, request, redirect, url_for, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from keras.models import load_model
import numpy as np
import cv2
import urllib.request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/files/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        print(file.filename)
        path = 'static/files/' + filename
        print(path)

        digit, confidence = predict_digit(path)
        
        flash("The digit you uploaded is the number: " + str(digit) + " and has a " + str(int(confidence * 100)) + "% confidence rate.")
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)

model = load_model('mnist.h5')

def predict_digit(img):
    img = cv2.imread(img)[:,:,0]
    img = np.invert(np.array([img]))
    img = img.reshape(1,28,28,1)
    img = img/255.0

    res = model.predict(img)[0]
    print(res)
    return np.argmax(res), max(res)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='files/' + filename), code =301)

if __name__ == "__main__":
    app.run(debug = True)