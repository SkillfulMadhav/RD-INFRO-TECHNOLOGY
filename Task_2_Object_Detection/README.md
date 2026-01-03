Task 2: Real-Time Object Detection System

Description
This project implements a real-time object detection system for security and surveillance use cases. The system captures live video input and detects multiple objects in real time using a deep learning–based object detection model.

Tools & Technologies

Python

OpenCV

YOLOv8 (Ultralytics)

Approach

Real-time video data is acquired using a webcam or video stream.

Each frame is preprocessed by resizing and formatting for inference.

A pre-trained YOLOv8 model is used to detect and classify objects.

Detected objects are highlighted with bounding boxes and confidence scores.

The system runs in real time using OpenCV’s video processing loop.

How to Run

Install dependencies:

pip install -r requirements.txt


Run the detection script:

python detect.py


Press q to exit the video window.

Optimization (Conceptual)
The system can be further optimized using inference engines such as TensorRT or OpenVINO to improve performance and reduce latency on edge devices.