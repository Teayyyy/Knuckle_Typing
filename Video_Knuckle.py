import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp

"""
RUN this to see the result of the model
"""

# load model from Training_Model1_23_6_10
knuckle_model = tf.keras.models.load_model('Training_Model1_23_6_10')

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam for Hand Gesture Python project
cap = cv2.VideoCapture(0)

count = 0

while True:
    _, frame = cap.read()
    x, y, z = frame.shape

    # frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    # get hand landmark
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        count += 1
        landmarks = []
        norm_landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
                mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
                norm_landmarks.append(lm.x)
                norm_landmarks.append(lm.y)
        if count >= 15:
            # prepare to predict
            norm_landmarks = np.array(norm_landmarks)
            landmark_tensor = tf.convert_to_tensor(norm_landmarks.reshape(1, 42), dtype=tf.float32)
            landmark_tensor = tf.expand_dims(landmark_tensor, axis=0)
            # predict
            prediction = knuckle_model.predict(landmark_tensor)
            # print(prediction * 100)
            # print(np.argmax(prediction))
            # 根据输出print出预测出的最大值的索引
            print(np.argmax(prediction))
            count = 0

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
