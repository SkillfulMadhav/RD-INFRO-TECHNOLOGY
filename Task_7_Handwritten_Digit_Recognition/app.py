import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

model = load_model("model/mnist_cnn.keras")

st.title("Handwritten Digit Recognition")

canvas = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data[:, :, 0]
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img)
        st.success(f"Predicted Digit: {np.argmax(pred)}")
