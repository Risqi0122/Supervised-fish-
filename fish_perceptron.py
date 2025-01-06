import streamlit as st
import pickle
import numpy as np

# Memuat kembali model, scaler, dan encoder menggunakan pickle
with open('perceptron_fish.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler_fish_Perseptron.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('label_encoder_fish_Perseptron.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)


# Judul Aplikasi
st.title('Prediksi Jenis Buah')

# Input untuk setiap fitur buah
diameter = st.number_input('Diameter:', min_value=0.0)
weight = st.number_input('Weight:', min_value=0.0)
red = st.number_input('Red:', min_value=0.0)
green = st.number_input('Green:', min_value=0.0)
blue = st.number_input('Blue:', min_value=0.0)

# Tombol untuk memprediksi spesies buah
if st.button('Prediksi Buah'):
    features = np.array([[diameter, weight, red, green, blue]])
    # Melakukan prediksi menggunakan model Perceptron
    species_encoded = model.predict(features)[0]
    # Mengembalikan hasil prediksi ke label asli menggunakan encoder
    species = encoder.inverse_transform([species_encoded])[0]
    # Menampilkan hasil prediksi
    st.success(f'Spesies yang Diprediksi: {species}')
