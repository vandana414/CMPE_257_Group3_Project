import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 #pip install opencv-python
import random
import pickle

CATEGORIES = ["positive","negative"]

def create_training_data(IMG_SIZE,DATA_DIR):
    training_data=[]
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        path = os.path.join(DATA_DIR,category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
    return training_data
    
def createPickle(Data,IMG_SIZE,filename):
    X=[]
    y=[]
    random.shuffle(Data)
    for features,label in Data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)

    pickle_out=open("pickle/X_"+filename+".pickle","wb")
    pickle.dump(X,pickle_out)
    pickle_out.close()

    pickle_out=open("pickle/y_"+filename+".pickle","wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()
    
training_data1 = create_training_data(224,"C:/Users/Vandana/Desktop/cmpe 257/CMPE_257_final/data/potholes_dataset_vandana/train")
createPickle(training_data1,224,str(len(training_data1)))
training_data2 = create_training_data(224,"C:/Users/Vandana/Desktop/cmpe 257/CMPE_257_final/data/potholes_dataset_manisha/train")
training_data3 = create_training_data(224,"C:/Users/Vandana/Desktop/cmpe 257/CMPE_257_final/data/pothole_data_tejal/train")
training_data =training_data1
training_data.extend(training_data2)
createPickle(training_data,224,str(len(training_data)))
training_data.extend(training_data3)
createPickle(training_data,224,str(len(training_data)))
training_data4 = create_training_data(224,"C:/Users/Vandana/Desktop/cmpe 257/CMPE_257_final/data/team_4_data/train")
training_data.extend(training_data4)
createPickle(training_data,224,str(len(training_data)))
training_data5 = create_training_data(224,"C:/Users/Vandana/Desktop/cmpe 257/CMPE_257_final/data/team_8/train")
training_data.extend(training_data5)
createPickle(training_data,224,str(len(training_data)))