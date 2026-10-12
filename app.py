import streamlit as st
import joblib
import numpy as np

# 1. Load the model saved in Task 4
try:
    model = joblib.load('iris_classifier_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'iris_classifier_model.pkl' is in the same directory.")
    st.stop()

st.title("Iris Flower Classifier 🌸")
st.write("Adjust the sliders below to predict the species of the Iris flower.")

# 2. Create interactive input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# 3. Interactive prediction display
if st.button("Predict Species"):
    # Format the inputs into a 2D array for the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Map the numerical prediction to the actual species name
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species_map.get(prediction[0], "Unknown")
    
    st.success(f"The predicted species is: **{predicted_species}**")
