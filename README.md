# 👗🔍 Fashion Dress Similarity Search Web App

This is a Flask-based web application that allows users to upload an image of a fashion item (e.g., a dress) and returns visually similar items from a predefined dataset. It leverages deep learning for feature extraction and cosine similarity for search, enabling an intuitive visual product discovery experience.

---

## 🎯 Project Aim

To build an intelligent fashion image search system that:
- Identifies visually similar fashion items based on image content.
- Leverages deep learning for robust feature extraction (using EfficientNetB0).
- A training module that allows users to upload a batch of 10 images every time to dynamically expand and enhance the searchable fashion image database.
- Enables both image-based and text-based product search functionalities.

---
<!-- 🌟 Project Preview Screenshots Section 🌟 -->

<!-- 🖼️ First Row: Main Training Page for Incremental Model -->
<!-- This screen shows how users can upload 10 images at a time to train the model incrementally (to handle memory efficiently). -->
| **Train Model Page** |
|:---------------------:|
| ![Train Model Page](images/home_screen.png) |

<!-- 🧍‍♂️🧍‍♀️🧑‍🤝‍🧑 Second Row: Category Sections -->
<!-- These three screens demonstrate product filtering by category: Men, Women, and Unisex. Each button loads the relevant section. -->
| **Men's Section** | **Women's Section** | **Unisex Section** |
|:------------------:|:--------------------:|:-------------------:|
| ![Men's Section](images/men_screen.png) | ![Women's Section](images/women_screen.png) | ![Unisex Section](images/unisex_screen.png) |

<!-- 🔍📷 Third Row: Search Capabilities -->
<!-- These screens show the app's search features: 
     1️⃣ Text-based search to retrieve matching products.
     2️⃣ Image-based search to find top-5 visually similar items. -->
| **Text-Based Search** | **Image-Based Search** |
|:----------------------:|:-----------------------:|
| ![Text Search](images/text_search_screen.png) | ![Image Search](images/image_search_screen.png) |

---

## 🚀 Features

- 🖼 Upload an image of a fashion item (dress or similar).
- ⚙️ Automatically extract deep visual features using a pre-trained EfficientNetB0 model.
- 🔍 Perform similarity search across the dataset using cosine similarity.
- 🎯 Display top 5 visually similar results to the uploaded image.
- 🔤 Filter search results using gender and product text query.

---

## 📁 Project Structure

```
fashion-image-search/
│
├── app.py                      # Main Flask application
├── styles.csv                  # Product metadata (IDs, gender, names, etc.)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── templates/                  # HTML templates for UI
│   ├── home.html               # Upload & training page
│   └── search.html             # Search results display
│
├── static/
│   ├── uploads/                # Temporary user-uploaded images
│   └── valid_images/           # Dataset images used for similarity comparison
│
├── model/
│   ├── features                # Saved image feature vectors (auto-generated)
│   └── image_ids               # Corresponding image file paths (auto-generated)
```

---

## 🛠️ Tech Stack

### **Frontend:**
- HTML
- CSS (via Flask Jinja templates)

### **Backend:**
- Python
- Flask

### **Machine Learning:**
- TensorFlow (EfficientNetB0 for feature extraction)
- Scikit-learn (cosine similarity)

### **Data Processing:**
- NumPy
- Pandas
- Pillow (PIL)

---

## 🚀 How to Run the Application

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the Flask app**:
```bash
python app.py
```

3. **Open in browser**:
```
http://127.0.0.1:5000/
```

---

## 📸 Example Use Case

1. **Training Mode**: Upload a batch of 10 fashion item images to train the system.
2. **Search Mode**:
   - Upload a query image to find visually similar fashion items.
   - Or, use the search form to filter by gender and/or product name.

---

## 🎉 **Thank You!**

Feel free to explore, fork, or enhance the application. Contributions are welcome!
