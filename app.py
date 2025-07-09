import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Deteksi Pneumonia dari X-ray")

model = tf.keras.models.load_model("model_xray.h5")

uploaded_file = st.file_uploader("Upload gambar X-ray (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Gambar X-ray', use_column_width=True)
    
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    
    st.write(f"### Hasil Prediksi: {label}")
    st.write(f"Confidence Score: {prediction:.2f}")