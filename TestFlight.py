# import necessary packages for hand gesture recognition project

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.python.keras.models import load_model


# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('Assets/gesture.names', 'r')
class_names = f.read().split('\n')
f.close()

print(class_names)

# Initialize the webcam for Hand Gesture Python project
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, z = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # process hand keypoints
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    # get hand landmark prediction
    result = hands.process(frame_rgb)

    class_name = ''
    #post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
                # drawing landmarks on frames
                mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()