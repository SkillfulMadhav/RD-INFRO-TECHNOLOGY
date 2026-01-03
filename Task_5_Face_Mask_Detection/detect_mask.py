import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("Loading model...")
model = load_model("model/mask_detector.keras")
print("Model loaded")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not accessible")
    exit()

print("Webcam started")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    face = cv2.resize(frame, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    mask, no_mask = model.predict(face)[0]

    label = "Mask" if mask > no_mask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
