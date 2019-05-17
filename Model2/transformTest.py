import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

class Transform:
    def create_test_data(IMG_SIZE,DATA_DIR):
        test_data=[]
        for img in os.listdir(DATA_DIR):
            try:
                img_array = cv2.imread(os.path.join(DATA_DIR,img))
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                test_data.append([new_array])
            except Exception as e:
                pass
        return test_data
        
    def createArray(Data,IMG_SIZE):
        X=[]
        for features in Data:
            X.append(features)

        X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
        return X
    