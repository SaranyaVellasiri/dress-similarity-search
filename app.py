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

# Load styles.csv
styles_df = pd.read_csv('Styles1.csv')
# Load model
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# File paths
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

        image_id = filename.split('.')[0]
        ids.append(image_id)

    if os.path.exists(FEATURES_FILE) and os.path.exists(IDS_FILE):
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

    return render_template('home.html', trained=True)

@app.route('/search', methods=['GET', 'POST'])
def search():
    results = []
    selected_gender = None

    if request.method == 'POST':
        gender = request.form.get('category')
        query_text = request.form.get('text_query')
        query_image = request.files.get('query')

        df = styles_df.copy()

        if gender:
            selected_gender = gender
            df = df[df['gender'].str.lower() == gender.lower()]

        if query_image and os.path.exists(FEATURES_FILE) and os.path.exists(IDS_FILE):
            path = os.path.join(app.config['UPLOAD_FOLDER'], query_image.filename)
            query_image.save(path)
            query_feat = extract_features(path)

            with open(FEATURES_FILE, 'rb') as f:
                features = pickle.load(f)
            with open(IDS_FILE, 'rb') as f:
                ids = pickle.load(f)

            similarities = cosine_similarity([query_feat], features)[0]
            top_indices = similarities.argsort()[::-1][:5]
            top_ids = [ids[i] for i in top_indices]

            for img_id in top_ids:
                img_path = f'valid_images/{img_id}.jpg'
                full_path = os.path.join('static', img_path)
                if os.path.exists(full_path):
                    name_row = styles_df[styles_df['id'] == int(img_id)]
                    product_name = name_row['productDisplayName'].values[0] if not name_row.empty else "Unknown"
                    results.append({'image': img_path, 'name': product_name})

        elif query_text or gender:
            if query_text:
                df = df[df['productDisplayName'].str.contains(query_text, case=False, na=False)]

            id_name_pairs = df[['id', 'productDisplayName']].dropna().head(100)
            for _, row in id_name_pairs.iterrows():
                img_id = str(row['id'])
                img_path = f'valid_images/{img_id}.jpg'
                full_path = os.path.join('static', img_path)
                if os.path.exists(full_path):
                    results.append({'image': img_path, 'name': row['productDisplayName']})

    return render_template('search.html', similar_results=results, selected_gender=selected_gender)
if __name__ == '__main__':
    app.run()

