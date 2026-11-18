# Clasificador de piedra, papel o tijera

Aplicación web con **Streamlit** y **ONNX Runtime** que clasifica imágenes
en `paper`, `rock` o `scissors`.

## Origen del modelo

- **NB04:** entrenamiento de la CNN.
- **NB07:** exportación del modelo original a ONNX.
- **NB08:** aplicación y despliegue.

## Contrato de inferencia

Imagen RGB, tamaño 128 × 128, tensor float32 normalizado dividiendo por 255.
Orden de las salidas: `paper`, `rock`, `scissors`.

## Archivos

- `app.py`: interfaz e inferencia.
- `modelo_cnn.onnx`: CNN exportada desde NB07.
- `class_names.json`: categorías y su orden.
- `model_contract.json`: contrato de entrada.
- `requirements.txt`: bibliotecas de ejecución.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

Subir estos archivos a un repositorio GitHub y seleccionar `app.py`
como archivo principal en Streamlit Community Cloud.

**Nota:** la confianza softmax no garantiza exactitud.
