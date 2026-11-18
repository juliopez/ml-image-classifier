
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Piedra, papel o tijera", page_icon="✋")
st.title("Piedra, papel o tijera")
st.write("Clasifica una imagen con la CNN del NB04, exportada a ONNX en NB07.")

@st.cache_resource
def cargar_modelo():
    return ort.InferenceSession(str(BASE_DIR / "modelo_cnn.onnx"),
                                providers=["CPUExecutionProvider"])

@st.cache_data
def cargar_metadatos():
    with open(BASE_DIR / "class_names.json", encoding="utf-8") as archivo:
        clases = json.load(archivo)
    with open(BASE_DIR / "model_contract.json", encoding="utf-8") as archivo:
        contrato = json.load(archivo)
    if clases != ["paper", "rock", "scissors"] or contrato["clases"] != clases:
        raise ValueError("El orden de clases no coincide con el NB04.")
    if contrato["input_shape"] != [1, 128, 128, 3]:
        raise ValueError("El tamaño de entrada debe ser 128 × 128 RGB.")
    return clases, contrato

try:
    session = cargar_modelo()
    class_names, contrato = cargar_metadatos()
    entrada_info = session.get_inputs()[0]
    salida_info = session.get_outputs()[0]
    if list(entrada_info.shape[1:]) != [128, 128, 3] or salida_info.shape[-1] != 3:
        raise ValueError("La forma del modelo ONNX es incompatible con el contrato.")
except Exception as exc:
    st.error(f"No se pudo cargar el modelo o sus metadatos: {exc}")
    st.stop()

def predecir(imagen):
    imagen = imagen.convert("RGB").resize((128, 128))
    x = np.asarray(imagen, dtype=np.float32)[None, ...] / 255.0
    probs = session.run([salida_info.name], {entrada_info.name: x})[0][0]
    indice = int(np.argmax(probs))
    return indice, float(probs[indice]), probs

archivo = st.file_uploader("Selecciona una imagen PNG o JPG",
                           type=["png", "jpg", "jpeg"])
if archivo is not None:
    try:
        imagen = Image.open(archivo)
        imagen.load()
        st.image(imagen, caption="Imagen seleccionada", use_container_width=True)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("El archivo no es una imagen válida.")
        st.stop()

    if st.button("Clasificar", type="primary"):
        indice, confianza, probabilidades = predecir(imagen)
        st.subheader("Resultado")
        st.write(f"**Clase predicha:** {class_names[indice]}")
        st.write(f"**Confianza:** {confianza:.2%}")
        if confianza < 0.60:
            st.warning("Confianza inferior al 60 %: interpreta el resultado con cautela.")
        tabla = pd.DataFrame({"Clase": class_names,
                              "Probabilidad": probabilidades})
        st.subheader("Probabilidades por clase")
        st.dataframe(tabla.sort_values("Probabilidad", ascending=False),
                     hide_index=True, use_container_width=True)
        st.bar_chart(tabla.set_index("Clase")["Probabilidad"])
        st.caption("La confianza softmax no garantiza que la predicción sea correcta.")
