import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp

# load model from Training_Model1_23_6_10
knuckle_model = tf.keras.models.load_model('../Training_Model1_23_6_10')

# load images
img_path = 'test_hands'
img = cv2.imread(img_path + '/4.jpg')
print(img.shape)

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# process hand keypoints
frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
# get hand landmark prediction
result = hands.process(frame_rgb)

print(result.multi_hand_landmarks)
# make an array contains x and y of landmarks

norm_landmarks = []
if result.multi_hand_landmarks:
    for handslms in result.multi_hand_landmarks:
        for lm in handslms.landmark:
            norm_landmarks.append(lm.x)
            norm_landmarks.append(lm.y)
            # drawing landmarks on frames


norm_landmarks = np.array(norm_landmarks)
print(norm_landmarks.shape)

# prepare to put in model
landmark_tensor = tf.convert_to_tensor(norm_landmarks.reshape(1, 42), dtype=tf.float32)
landmark_tensor = tf.expand_dims(landmark_tensor, axis=0)

# predict
prediction = knuckle_model.predict(landmark_tensor)
print(prediction * 100)
print(np.argmax(prediction))