import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

import platform

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600&family=Nunito:wght@400;600&display=swap');

.stApp {
    background-color: #FFF9EC;
    background-image: radial-gradient(#D7C9A8 1.3px, transparent 1.3px);
    background-size: 24px 24px;
}

h1, h2, h3 {
    font-family: 'Fredoka', sans-serif !important;
    color: #2F6F9F !important;
}

h2 {
    display: inline-block;
    background-color: #CDE9F9;
    border: 2.5px solid #2F6F9F;
    border-radius: 18px;
    padding: 10px 20px !important;
    box-shadow: 4px 4px 0px #2F6F9F;
    transform: rotate(-0.8deg);
    font-size: 1.4rem !important;
}

p, li, label, div[data-testid="stMarkdownContainer"], .stMarkdown {
    font-family: 'Nunito', sans-serif !important;
    color: #3D4A55 !important;
}

section[data-testid="stSidebar"] {
    background-color: #EAF5FC;
    border-right: 3px dashed #7DB9DE;
}

div[data-testid="stImage"] {
    position: relative;
    display: inline-block;
}
div[data-testid="stImage"] img {
    border: 8px solid #FFFFFF;
    border-radius: 6px;
    box-shadow: 3px 4px 10px rgba(0,0,0,0.18);
    transform: rotate(1.2deg);
}
div[data-testid="stImage"]::before {
    content: "";
    position: absolute;
    top: -10px;
    left: 50%;
    width: 90px;
    height: 24px;
    margin-left: -45px;
    background-color: rgba(125,185,222,0.75);
    transform: rotate(-3deg);
    z-index: 2;
}

div[data-testid="stCameraInput"] {
    background-color: #FFFFFF;
    border: 3px dashed #7DB9DE;
    border-radius: 16px;
    padding: 12px;
}

.stButton button, div[data-testid="stCameraInput"] button {
    font-family: 'Fredoka', sans-serif !important;
    background-color: #7DB9DE !important;
    color: #FFFFFF !important;
    border: 2.5px solid #2F6F9F !important;
    border-radius: 14px !important;
    box-shadow: 3px 3px 0px #2F6F9F;
}
</style>
""", unsafe_allow_html=True)

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Recordatorio de Hidratación 💧")
#st.write("Versión de Python:", platform.python_version())
image = Image.open('OIG5.jpg')
st.image(image, width=350)
with st.sidebar:
    st.subheader("Muestra tu botella de agua a la cámara: un modelo entrenado en Teachable Machine la identifica y te recuerda mantenerte hidratado mientras estudias o programas.")
img_file_buffer = st.camera_input("Muéstrame tu botella de agua")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.header('No veo tu botella 💧 ¡Es hora de tomar agua! (confianza: '+str( prediction[0][0]) +')')
    if prediction[0][1]>0.5:
      st.header('¡Botella detectada! Bien hecho, sigue hidratándote 💧 (probabilidad: '+str( prediction[0][1])+')')
    #if prediction[0][2]>0.5:
    # st.header('Derecha, con Probabilidad: '+str( prediction[0][2]))
