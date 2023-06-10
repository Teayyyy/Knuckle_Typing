# import necessary packages for hand gesture recognition project

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.python.keras.models import load_model

"""
This don't work, don't know what happened
"""

# load model
knuckle_model = tf.keras.models.load_model('Training_Model1_23_6_10')

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


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
        norm_landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
                norm_landmarks.append(lm.x)
                norm_landmarks.append(lm.y)
                # drawing landmarks on frames
                mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

        # transform list into tensor shape=(None, 1, 42)
        norm_landmarks = np.array(norm_landmarks)
        landmark_tensor = tf.convert_to_tensor(norm_landmarks.reshape(1, 42), dtype=tf.float32)
        landmark_tensor = tf.expand_dims(landmark_tensor, axis=0)

        print(landmark_tensor)
        # predictions = knuckle_model(landmark_tensor)
        predictions = knuckle_model.predict(landmark_tensor)
        print(predictions)

        # print(pred_knuckle.index(1))

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()