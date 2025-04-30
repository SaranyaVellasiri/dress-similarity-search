👗🔍 Fashion Dress Similarity Search Web App

This is a Flask-based web application that allows users to upload an image of a fashion item (e.g., a dress) and returns visually similar items from a predefined dataset. It leverages deep learning for feature extraction and cosine similarity for search, enabling an intuitive visual product discovery experience.

------------------------------------------------------------

🎯 AIM:  
To build a web-based application that helps users find visually similar fashion items (dresses) by uploading an image. This leverages deep learning feature extraction and similarity search to power fashion discovery.

🚀 Features:
- Upload an image of a dress.
- Automatically extract visual features using a pre-trained deep learning model (EfficientNetB0).
- Perform similarity search from the dataset using cosine similarity.
- Return visually similar fashion items with display.

------------------------------------------------------------

📁 Project Structure:

Fastion-Dress-similarity-search/
│
├── app.py                  # Main Flask application
├── styles.csv              # Metadata or label information for fashion items
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── templates/              # HTML Templates
│   ├── home.html           # Upload interface
│   └── search.html         # Display results
│
├── static/
│   ├── uploads/            # User-uploaded images
│   └── valid_images/       # Fashion images dataset
│
├── model/
│   ├── features/           # Saved extracted features (auto-generated)
│   └── image_ids/          # Corresponding image IDs (auto-generated)

------------------------------------------------------------

🛠️ Tech Stack:

Frontend:
- HTML
- CSS (via Flask templates)

Backend:
- Python
- Flask

Machine Learning:
- TensorFlow (EfficientNetB0 for feature extraction)
- Scikit-learn (Cosine similarity)

Data Processing:
- NumPy
- Pandas
- Pillow
