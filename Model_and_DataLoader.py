import pandas
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
import ast

"""
This file contains:
    * Structure of network
    * Data loader for csv
"""
class CSV_Dense_Model(Model):
    def __init__(self):
        """
        Input layer: 21 * 2 = 42
            Result of hands landmarks generated from Mediapipe are 21 key points
        Output layer: 12
            Knuckle of four fingers contains 12
        """
        super().__init__()
        self.den0 = layers.Dense(42, activation=tf.nn.sigmoid)
        self.drop1 = layers.Dropout(0.3)
        self.den1 = layers.Dense(22, activation=tf.nn.sigmoid)
        self.drop2 = layers.Dropout(0.2)
        self.den2 = layers.Dense(16, activation=tf.nn.sigmoid)
        self.drop3 = layers.Dropout(0.2)
        self.den_output = layers.Dense(12, activation=tf.nn.softmax)

    def call(self, inputs, **kwargs):
        inputs = self.den0(inputs)
        inputs = self.drop1(inputs)
        inputs = self.den1(inputs)
        inputs = self.drop2(inputs)
        inputs = self.den2(inputs)
        inputs = self.drop3(inputs)
        return self.den_output(inputs)



class Data_Loader:
    @staticmethod
    # Verified
    def load_csv():
        print("Loading data from csv...")
        # data_and_label = pd.read_csv('Assets/all_hands_with_landmark1.csv')
        # use normalized data
        data_and_label = pd.read_csv('Assets/norm_all_hands_with_landmark_normalized.csv')
        print("Length of data: ", len(data_and_label))
        print("Converting landmarks...")  # from str to literal eval: list
        data = data_and_label['landmarks'].apply(ast.literal_eval)
        label = data_and_label['label'].to_numpy(dtype=np.uint8)
        new_data = []
        for i in range(len(data)):
            # print(type(data[i]))
            # print(data[i])
            temp = []
            for landmark in data[i]:
                for ele in landmark:
                    temp.append(ele)
            new_data.append(temp)
        data = new_data

        print("Shuffling data...", end="")
        combined = np.column_stack([data, label])
        np.random.shuffle(combined)
        print("Done!")
        # split
        shuffled_data = combined[:, :-1]
        shuffled_label = combined[:, -1:]

        # squeeze list()
        # shuffled_data = [data[0] for data in shuffled_data]
        # temp = []
        # for data in shuffled_data:
        #     temp.append(data[0])
        # shuffled_data = np.array(temp)

        shuffled_data = shuffled_data.squeeze()
        print("shape: ", shuffled_data.shape)
        print("data type: ", type(shuffled_data[0]))
        print("label type: ", type(shuffled_label[0]))
        print("total data: ", len(shuffled_data))
        print("total label: ", len(shuffled_label))
        return shuffled_data, shuffled_label


    """
    To normalize the landmark in same 'direction' (actually I don't know which direction XD)
        set first point as origin point in default
    """
    @staticmethod
    def normalize_landmark_direction(landmarks):
        # 选择参考点的索引（这里假设参考点是第一个关键点）
        landmarks = np.array(landmarks)
        reference_point_idx = 0

        # 获取参考点的坐标
        reference_point = landmarks[reference_point_idx]

        # 计算方向向量
        direction_vectors = landmarks - reference_point

        # 计算方向向量的长度
        direction_lengths = np.linalg.norm(direction_vectors, axis=1)

        # 规范化方向向量
        normalized_directions = direction_vectors / direction_lengths[:, np.newaxis]

        norm_list = normalized_directions.tolist()
        norm_list[0] = [0, 0]

        # 返回规范化后的landmark
        return norm_list
