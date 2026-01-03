Handwritten Digit Recognition System
Overview

This project implements a handwritten digit recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset.
Users can draw digits in a web interface, and the trained model predicts the digit in real time.

The project demonstrates the full workflow of training a deep learning model and integrating it into an interactive web application.

Features

CNN model trained on the MNIST dataset

Real-time digit prediction

Interactive drawing canvas

Simple and lightweight UI using Streamlit

Tech Stack

Python

TensorFlow / Keras

OpenCV

Streamlit

MNIST Dataset

Project Structure
Task_7_Handwritten_Digit_Recognition/
│
├── app.py                # Streamlit application
├── train.py              # Model training script
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── model/
    └── mnist_cnn.keras   # Trained CNN model

Setup Instructions
1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Train the model
python train.py


This will train a CNN on the MNIST dataset and save the model inside the model/ directory.

Running the Application
streamlit run app.py


Open the browser at http://localhost:8501

Draw a digit on the canvas

The model predicts the digit instantly

Notes

This project is for educational purposes only.

Model accuracy depends on drawing clarity.

The canvas feature uses the streamlit-drawable-canvas component.

Output

Trained CNN model (mnist_cnn.keras)

Web-based handwritten digit recognition interface

Author

Madhav Tiwary
B.Tech CSE