import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Load class names
classNames = []
classFile = 'coco.names'  # Assuming coco.names contains class names in each line
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load pre-trained DNN model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  # Set input size
net.setInputScale(1.0 / 127.5)  # Set input scale
net.setInputMean((127.5, 127.5, 127.5))  # Set mean subtraction (127.5, 127.5, 127.5)
net.setInputSwapRB(True)  # Set input swap RB channels (OpenCV uses BGR)

# Set parameters for object detection
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.2  # Non-Max Suppression threshold

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    classes, scores, boxes = net.detect(frame, confThreshold, nmsThreshold)

    # Process detections
    if len(classes) > 0:
        for classId, confidence, box in zip(classes.flatten(), scores.flatten(), boxes):
            className = classNames[classId-1]  # Get the class name
            score = confidence  # Confidence score of the detection
            left, top, width, height = box  # Bounding box coordinates

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (left, top), (left+width, top+height), color=(0, 255, 0), thickness=3)
            cv2.putText(frame, f'{className.upper()} [{score:.2f}]', (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
