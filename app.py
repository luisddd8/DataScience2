import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar el modelo y los componentes
@st.cache_resource
def load_components():
    model = joblib.load('vehicle_classifier_svm.pkl')
    scaler = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, imputer, label_encoder

model, scaler, imputer, label_encoder = load_components()

# Clases de vehículos
vehicle_classes = {
    0: 'van',
    1: 'saab',
    2: 'bus',
    3: 'opel'
}

# Configuración de la página
st.set_page_config(page_title="Clasificador de Vehículos", layout="wide")
st.title("Clasificador de Vehículos usando SVM")

# Descripción
st.markdown("""
Esta aplicación predice el tipo de vehículo (van, saab, bus, opel) basado en sus características.
Ingresa los valores de las características del vehículo y haz clic en 'Predecir'.
""")

# Formulario para ingresar características
with st.form("vehicle_features"):
    st.header("Ingresa las características del vehículo")
    
    # Dividir en columnas para mejor organización
    col1, col2, col3 = st.columns(3)
    
    with col1:
        compactness = st.number_input("Compactness", min_value=0.0, max_value=200.0, value=100.0)
        circularity = st.number_input("Circularity", min_value=0.0, max_value=200.0, value=100.0)
        distance_circularity = st.number_input("Distance Circularity", min_value=0.0, max_value=200.0, value=100.0)
        radius_ratio = st.number_input("Radius Ratio", min_value=0.0, max_value=200.0, value=100.0)
    
    with col2:
        pr_axis_aspect_ratio = st.number_input("PR Axis Aspect Ratio", min_value=0.0, max_value=200.0, value=100.0)
        max_length_aspect_ratio = st.number_input("Max Length Aspect Ratio", min_value=0.0, max_value=200.0, value=100.0)
        scatter_ratio = st.number_input("Scatter Ratio", min_value=0.0, max_value=200.0, value=100.0)
        elongatedness = st.number_input("Elongatedness", min_value=0.0, max_value=200.0, value=100.0)
    
    with col3:
        pr_axis_rectangularity = st.number_input("PR Axis Rectangularity", min_value=0.0, max_value=200.0, value=100.0)
        max_length_rectangularity = st.number_input("Max Length Rectangularity", min_value=0.0, max_value=200.0, value=100.0)
        scaled_variance = st.number_input("Scaled Variance", min_value=0.0, max_value=200.0, value=100.0)
        scaled_variance_1 = st.number_input("Scaled Variance 1", min_value=0.0, max_value=200.0, value=100.0)
    
    # Más características
    scaled_radius_of_gyration = st.number_input("Scaled Radius of Gyration", min_value=0.0, max_value=200.0, value=100.0)
    scaled_radius_of_gyration_1 = st.number_input("Scaled Radius of Gyration 1", min_value=0.0, max_value=200.0, value=100.0)
    skewness_about = st.number_input("Skewness About", min_value=0.0, max_value=200.0, value=100.0)
    skewness_about_1 = st.number_input("Skewness About 1", min_value=0.0, max_value=200.0, value=100.0)
    skewness_about_2 = st.number_input("Skewness About 2", min_value=0.0, max_value=200.0, value=100.0)
    hollows_ratio = st.number_input("Hollows Ratio", min_value=0.0, max_value=200.0, value=100.0)
    
    submit_button = st.form_submit_button("Predecir")

# Procesar la predicción
if submit_button:
    # Crear array con las características
    features = np.array([
        compactness, circularity, distance_circularity, radius_ratio,
        pr_axis_aspect_ratio, max_length_aspect_ratio, scatter_ratio, elongatedness,
        pr_axis_rectangularity, max_length_rectangularity, scaled_variance, scaled_variance_1,
        scaled_radius_of_gyration, scaled_radius_of_gyration_1, skewness_about,
        skewness_about_1, skewness_about_2, hollows_ratio
    ]).reshape(1, -1)
    
    # Preprocesamiento
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    
    # Predicción
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)
    
    # Obtener nombre de la clase
    predicted_class = vehicle_classes[prediction[0]]
    
    # Mostrar resultados
    st.success(f"El vehículo predicho es: **{predicted_class.upper()}**")
    
    # Mostrar probabilidades
    st.subheader("Probabilidades por clase:")
    proba_df = pd.DataFrame({
        'Clase': [vehicle_classes[i] for i in range(len(vehicle_classes))],
        'Probabilidad': prediction_proba[0]
    })
    st.bar_chart(proba_df.set_index('Clase'))

# Información adicional
st.sidebar.header("Acerca de")
st.sidebar.info("""
Esta aplicación utiliza un modelo SVM para clasificar vehículos en 4 categorías:
- Van
- Saab
- Bus
- Opel

El modelo fue entrenado con el dataset Vehicle del repositorio Statlog.
""")
