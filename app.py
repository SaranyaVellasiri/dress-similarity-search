from flask import Flask, render_template, request
import os
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VALID_FOLDER'] = 'static/valid_images'
app.config['MODEL_FOLDER'] = 'model'

# Load CSV
styles_df = pd.read_csv('styles.csv')

# Load base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Paths
FEATURES_FILE = os.path.join(app.config['MODEL_FOLDER'], 'features')
IDS_FILE = os.path.join(app.config['MODEL_FOLDER'], 'image_ids')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VALID_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def extract_features(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    features = model.predict(img_array)[0]
    return features

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    if len(files) != 10:
        return render_template('home.html')

    features, ids = [], []

    for file in files:
        filename = file.filename
        filepath = os.path.join(app.config['VALID_FOLDER'], filename)
        file.save(filepath)

        feature_vector = extract_features(filepath)
        features.append(feature_vector)
        ids.append(f'valid_images/{filename}')

    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, 'rb') as f:
            existing_features = pickle.load(f)
        with open(IDS_FILE, 'rb') as f:
            existing_ids = pickle.load(f)
        features = np.vstack([existing_features, features])
        ids = existing_ids + ids

    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(np.array(features), f)
    with open(IDS_FILE, 'wb') as f:
        pickle.dump(ids, f)

    return render_template('home.html', trained=True, accuracy="N/A")

@app.route('/search', methods=['GET', 'POST'])
def search():
    results = []
    if request.method == 'POST':
        gender = request.form.get('category')
        query_text = request.form.get('text_query')
        query_image = request.files.get('query')

        if not os.path.exists(FEATURES_FILE) or not os.path.exists(IDS_FILE):
            return render_template('search.html', similar_results=[])

        with open(FEATURES_FILE, 'rb') as f:
            features = pickle.load(f)
        with open(IDS_FILE, 'rb') as f:
            ids = pickle.load(f)

        if query_image:
            path = os.path.join(app.config['UPLOAD_FOLDER'], query_image.filename)
            query_image.save(path)
            query_feat = extract_features(path)
            similarities = cosine_similarity([query_feat], features)[0]
            top_indices = similarities.argsort()[::-1][:5]
            results = [ids[i] for i in top_indices]

        elif query_text or gender:
            df = styles_df.copy()
            if gender:
                df = df[df['gender'].str.lower() == gender.lower()]
            if query_text:
                df = df[df['productDisplayName'].str.contains(query_text, case=False, na=False)]
            results = [f'valid_images/{id}.jpg' for id in df['id'].astype(str).values[:100]
                       if os.path.exists(os.path.join('static', 'valid_images', f'{id}.jpg'))]

    return render_template('search.html', similar_results=results)

if __name__ == '__main__':
    app.run(
        )
