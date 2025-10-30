from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from gradcam_utils import make_gradcam_heatmap, save_and_display_gradcam

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
MODEL_PATH = "brain_tumor_model.h5"
model = load_model(MODEL_PATH)
CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)
GRAD_CAM_LAYER_NAME = 'Conv_1'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    pred_class = CATEGORIES[pred_index]
    confidence = float(predictions[0][pred_index])

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, GRAD_CAM_LAYER_NAME, pred_index)
    cam_image = save_and_display_gradcam(filepath, heatmap, IMG_SIZE)
    cam_path = os.path.join(app.config['UPLOAD_FOLDER'], f"cam_{filename}")
    cam_image.save(cam_path)

    return render_template('result.html',
                           filename=filename,
                           pred_class=pred_class,
                           confidence=round(confidence*100, 2),
                           cam_image=f"uploads/cam_{filename}")

if __name__ == '__main__':
    app.run(debug=True)
