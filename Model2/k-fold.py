import pandas as pd
from Model2 import Model2
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,accuracy_score

X=pickle.load(open("pickle/X_3021.pickle","rb"))
y=pickle.load(open("pickle/y_3021.pickle","rb"))



#K-fold validation
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index],X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model_val = Model2()
    model_val.trainModel(X_train,y_train)
    y_pred = model_val.predict(X_test)
    print("Accuracy: %f"%(accuracy_score(y_test,y_pred)))