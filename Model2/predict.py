import numpy as np
import pandas as pd
from transformTest import Transform
from Model2 import Model2
from collections import Counter
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score


    
if __name__ == '__main__':
    test_data = Transform.create_test_data(224,"C:/Users/Vandana/Desktop/cmpe 257/CMPE_257_final/data/test")
    X = Transform.createArray(test_data,224)
    
    #Predict using Logistic Regression trained model
    model = Model2()
    model.load_model("models/model2")
    y_pred = model.predict(X)
    print(y_pred)
    