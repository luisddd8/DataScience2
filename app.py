import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('best_svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('imputer.pkl', 'rb') as imputer_file:
    imputer = pickle.load(imputer_file)

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

st.title("Clasificador de Vehículos")
st.write("Este modelo predice el tipo de vehículo (van, saab, bus, opel) usando 18 atributos.")

# Crear inputs para los atributos
feature_names = [f"Atr{i+1}" for i in range(18)]
user_input = []

st.subheader("Ingresa los valores de los atributos:")
for name in feature_names:
    val = st.number_input(name, value=0.0)
    user_input.append(val)

if st.button("Predecir"):
    input_array = np.array([user_input])

    # Imputar y escalar
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)

    # Predecir
    prediction = model.predict(input_scaled)
    prediction_label = label_encoder.inverse_transform(prediction)

    st.success(f"El modelo predice que el vehículo es: **{prediction_label[0]}**")

    # Mostrar probabilidades
    probs = model.predict_proba(input_scaled)[0]
    for i, prob in enumerate(probs):
        st.write(f"{label_encoder.classes_[i]}: {prob:.2%}")

