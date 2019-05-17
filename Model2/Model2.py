import pickle
import numpy as np
import cv2
from sklearn.decomposition import PCA
from joblib import dump,load
from sklearn.ensemble import RandomForestClassifier

class Model2:
    #random forest classifier
    model = RandomForestClassifier(n_estimators=300, max_depth=10,random_state=0)
	#model = svm.SVC(gamma=0.0001,kernel='rbf')
    #model = LogisticRegression(C=0.001,penalty='l2',random_state=10,solver='liblinear',multi_class='ovr',class_weight='balanced')


    hog = cv2.HOGDescriptor()
    pca = PCA(n_components=250)

    def preprocess(self,img_array):
        R=[]
        for i in range(img_array.shape[0]):
            h=self.hog.compute(img_array[i])
            R.append(h.ravel())
        return R
        
    def fitPCA(self,train):
        self.pca.fit(train)
    
    def transformPCA(self,data):
        return self.pca.transform(data)

    def trainModel(self,train,labels):
        train = self.preprocess(train)
        print("Preprocessing Done")
        self.fitPCA(train)
        train = self.transformPCA(train)
        print("PCA Done")
        self.model.fit(train, labels)

    def predict(self,test):
        test = self.preprocess(test)
        test=self.transformPCA(test)
        pred = self.model.predict(test)
        return pred
    
    def save(self,f_name):
        dump(self.pca, f_name+'_pca.joblib')
        dump(self.model,f_name+'_model.joblib')
        print("Model saved")
        
    def load_model(self,f_name):
        self.pca = load(f_name+'_pca.joblib')
        self.model = load(f_name+'_model.joblib')
    

if __name__ == '__main__':
    X=pickle.load(open("pickle/X_391.pickle","rb"))
    y=pickle.load(open("pickle/y_391.pickle","rb"))
  
    #train and save the model
    model = Model2()
    model.trainModel(X,y)
    model.save("models/model_rf")
	#model.save("models/model_svm")
	#model.save("models/model_lr")
