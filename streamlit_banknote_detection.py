#!/usr/bin/env python
# coding: utf-8

# In[9]:


import io
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import base64
import numpy as np

# setting background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('image-7.jpg')

# importing model
MODEL_PATH = 'model100.h5'  # Replace with your TensorFlow model file
# importing class names
LABELS_PATH = 'label.txt'

# Load the model
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

# Load labels
def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories

# Image picker
def load_image():
    uploaded_file = st.file_uploader(label='Upload a banknote to test')
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((224, 224))  # Resize the image to 224x224
        st.image(image)
        return image
    else:
        return None


# Make predictions using the TensorFlow model
def predict(model, categories, image):
    image = image.resize((224, 224))  # Resize the image to match the model's input size
    image = np.array(image)  # Convert to NumPy array
    image = preprocess_input(image)  # Preprocess the image according to the model's requirements

    input_batch = tf.convert_to_tensor([image])

    predictions = model.predict(input_batch)
    top_categories = predictions[0].argsort()[-len(categories):][::-1]

    for i in top_categories:
        st.write(categories[i], predictions[0][i])

def main():
    st.title('Colombian Peso Banknote Detection')
    model = load_model(MODEL_PATH)
    categories = load_labels(LABELS_PATH)
    image = load_image()
    result = st.button('Predict image')
    
    if result and image is not None:
        st.write('Checking...')
        predict(model, categories, image)

if __name__ == '__main__':
    main()

