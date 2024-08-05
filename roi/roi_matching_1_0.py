import os
import pickle
import time

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

from roi.model_util import DeepModel


class ImageClassifier:
    def __init__(self):
        self.all_skus = {}
        self.model = DeepModel()
        self.predict_time = 0
        self.time_search = 0
        self.count_frame = 0
        self.top_k = 5

    def extract_features_from_img(self, cur_img):
        cur_img = cv2.resize(cur_img, (224, 224))
        img = preprocess_input(cur_img)
        img = np.expand_dims(img, axis=0)
        feature = self.model.extract_feature(img)
        return feature

    def predict(self, img):
        self.count_frame += 1
        before_time = time.time()
        target_features = self.extract_features_from_img(img)
        self.predict_time += time.time() - before_time
        max_distance = 0
        result_dish = 0

        for dish, features_all in self.all_skus.items():
            for features in features_all:
                cur_distance = self.model.cosine_distance(target_features, features)
                cur_distance = cur_distance[0][0]
                if cur_distance > max_distance:
                    max_distance = cur_distance
                    result_dish = dish
        print(self.predict_time, result_dish)
        return result_dish, max_distance

    def add_img(self, img_path, id_img):
        img = cv2.imread(img_path)
        cur_img = img
        feature = self.extract_features_from_img(cur_img)
        if id_img not in self.all_skus:
            self.all_skus[id_img] = []
        self.all_skus[id_img].append(feature)
        return feature

    def remove_by_id(self, id_img):
        if id_img in self.all_skus:
            self.all_skus.pop(id_img)

    def remove_all(self):
        self.all_skus.clear()

    def add_img_from_pickle(self, id_img, pickle_path):
        res = pickle.load(open(pickle_path, 'rb'))
        self.all_skus[id_img] = res

    def get_additional_info(self):
        json_res = {}
        json_res["Extract features, time"] = self.predict_time
        json_res["Find nearest, time"] = self.time_search
        json_res["Count frame"] = self.count_frame
        json_res["RPS"] = self.count_frame / (self.predict_time + self.time_search)
        return json_res


if __name__ == "__main__":
    test_images = 'roi_data'
    path_t = 'roi_test'
    performance_time = time.time()
    classifier = ImageClassifier()
    print("Инициализация ImageClassifier",  time.time() - performance_time)

    performance_time = time.time()
    test_paths = os.listdir(test_images)
    for f in test_paths:
        classifier.add_img(os.path.join(test_images, f), f)
    print("Добавление одного фото classifier.add_img", (time.time() - performance_time) / 5)

    performance_time = time.time()
    for data_img in os.listdir(path_t):
        frame = cv2.imread(os.path.join(path_t, data_img))
        name, dist = classifier.predict(frame)
    print("Обработка 1500 фото", time.time() - performance_time)