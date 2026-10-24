import streamlit as st
import pickle
import numpy as np

# 1. Model ko load karein
try:
    with open('starbucks_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'starbucks_model.pkl' file nahi mili. Pehle model file upload karein.")
    model = None

# 2. Web App ka Title aur Header
st.set_page_config(page_title="Starbucks Price Predictor", page_icon="☕", layout="centered")
st.title("☕ Starbucks Price Predictor App")
st.write("Decision Tree Regressor Model ka use karke prediction karein.")

st.divider()

# 3. User Input Fields (Aapke Decision Tree ke mutabiq)
st.subheader("Enter Details:")
quantity = st.number_input("Quantity:", min_value=1.0, max_value=100.0, value=1.0, step=1.0)
price = st.number_input("Base Price ($):", min_value=0.5, max_value=500.0, value=5.0, step=0.5)

st.divider()

# 4. Prediction Button
if st.button("Predict Target Value", type="primary"):
    if model is not None:
        # Features ko array me convert karein (Quantity aur Price)
        features = np.array([[quantity, price]])
        
        # Model se predict karein
        prediction = model.predict(features)
        
        # Output show karein
        st.success(f"🎯 Predicted Output Value: **${prediction[0]:.2f}**")
    else:
        st.error("Model load nahi ho saka, prediction cancel.")
