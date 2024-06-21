import web_app as st
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
import pickle

# Load pretrained model
@st.cache_data
def load_model():
    with open('digits_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

model = load_model()

# Function to make predictions
def predict_digit(features):
    prediction = model.predict(features)
    return prediction

# Main title and user input fields
st.title('Digit Recognition')
st.write('Enter the pixel values of a digit image (8x8) to predict the digit.')

# User input for pixel values (8x8 image)
features = st.text_area('Pixel values', '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 1, 0, 0, 0, 0, 4, 8, 8, 8, 8, 7, 0, 0, 0, 0, 5, 8, 8, 8, 8, 0, 0, 0, 0, 5, 8, 8, 8, 8, 0, 0, 0, 0, 6, 8, 8, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0')

# Convert input to numpy array and reshape to (1, 64)
features_list = [int(x) for x in features.split(',')]
features_array = np.array(features_list).reshape(1, -1)

# Prediction and display result
if st.button('Predict'):
    prediction = predict_digit(features_array)
    st.write(f'Predicted Digit: {prediction[0]}')