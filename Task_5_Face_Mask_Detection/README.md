# Face Mask Detection System

This project implements a real-time face mask detection system using deep learning and computer vision.

The system uses transfer learning with MobileNetV2 to classify whether a person is wearing a face mask or not from live webcam input.

## Tools and Technologies
- Python
- TensorFlow / Keras
- OpenCV
- MobileNetV2 (pretrained on ImageNet)

## Project Structure
- train_mask.py : Trains the mask detection model using a labeled dataset
- detect_mask.py : Runs real-time face mask detection using webcam
- experiments/data : Dataset containing with_mask and without_mask images
- model/mask_detector.keras : Trained model file

## How It Works
1. Images are preprocessed and resized to 224x224.
2. MobileNetV2 is used as the base model with frozen layers.
3. A custom classification head is trained for mask detection.
4. The trained model is used for real-time inference via OpenCV.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Train the model:
   python train_mask.py
3. Run real-time detection:
   python detect_mask.py

Press 'q' to exit the webcam window.

## Note
This project is for educational purposes only.
