import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelos
model = joblib.load('best_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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

