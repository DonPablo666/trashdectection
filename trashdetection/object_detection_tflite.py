import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load the TFLite model and allocate tensors
# Specify the full path to the TFLite model file
model_path = '/home/renalyn/trashdetection/model-1.tflite'

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the class names
class_names = ["di malata", "malata", "metal-can", "recycle"]

# Function to preprocess the frame for the model
def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame, (32, 32))

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()

    # Get the predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(predictions)
    label = class_names[index]
    confidence = predictions[0][index]

    # Display the label and confidence on the frame
    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
