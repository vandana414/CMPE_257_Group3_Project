import pickle
import numpy as np
import cv2
import random
import os
import cv2

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


#Load the images and convert to array
training_data1 = create_training_data(224,"C:/Users/Vandana/Desktop/cmpe 257/Project/Project/data/potholes_dataset_vandana/train")
createPickle(training_data1,224,str(len(training_data1)))

# Load image array
X=pickle.load(open("pickle/X_399.pickle","rb"))
y=pickle.load(open("pickle/y_399.pickle","rb"))

hog = cv2.HOGDescriptor()

from sklearn.decomposition import PCA
pca = PCA(n_components=250)

def preprocess(img_array):
    R=[]
    for i in range(img_array.shape[0]):
        h=hog.compute(img_array[i])
        R.append(h.ravel())
    return R


# Apply Kmeans to form 2 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=40,max_iter=400,init='k-means++',n_init=20)


# Pipeline to preprocess and fit the data
def fit(train):
    train = preprocess(train)
    print("Preprocessing Done")
    train = pca.fit_transform(train)
    print("PCA Done")
    kmeans.fit(train)


fit(X)

#Evaluate the v measure for the clusters formed
import sklearn.metrics as metrics
print("Kmeans Accuracy:")
print(metrics.v_measure_score(y, kmeans.labels_))

