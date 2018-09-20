# -*- coding: utf-8 -*-
"""
Created on Thu May  3 23:20:48 2018

@author: Anirudh
"""
"""
ANIRUDH SIVARAMAKRISHNAN 1001529484

"""
import pandas as pd
import numpy as np
from sklearn import svm
from collections import Counter
import math
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  linear_model
filename = "GenomeTrainXY.txt"
filename2 = "GenomeTestX.txt"
dataset = pd.read_csv(filename,sep=',');
dataset_test = pd.read_csv(filename2,sep=',')
print(dataset.shape);
dataset_trimmed = np.array(dataset.iloc[0:,0:])
test_X = np.array(dataset_test.iloc[0:,0:])
def f_anova(i):
    class_1_list = []
    class_1_list = list(dataset_trimmed[i][0:11])
    class_1_avg = np.mean(class_1_list)
    class_1_var = np.var(class_1_list)
    class_2_list = []
    class_2_list = list(dataset_trimmed[i][11:17])
    class_2_avg = np.mean(class_2_list)
    class_2_var = np.var(class_2_list)
    class_3_list = []
    class_3_list = list(dataset_trimmed[i][17:28])
    class_3_avg = np.mean(class_3_list)
    class_3_var = np.var(class_3_list)
    class_4_list = []
    class_4_list = list(dataset_trimmed[i][28:40])
    class_4_avg = np.mean(class_4_list)
    class_4_var = np.var(class_4_list)
    class_1_2_3_4_list = class_1_list + class_2_list + class_3_list + class_4_list
    class_1_2_3_4_avg = np.mean(class_1_2_3_4_list)
    sum_n = ((11*(class_1_avg - class_1_2_3_4_avg)**2) + (6*(class_2_avg - class_1_2_3_4_avg)**2) + (11*(class_3_avg - class_1_2_3_4_avg)**2) + (12*(class_4_avg - class_1_2_3_4_avg)**2))/3
    sum_d = ((11*class_1_var) + (6*class_2_var) + (11*class_3_var) + (12*class_4_var))/36
    f_score = sum_n/sum_d
    return f_score
fscore_list = []
for i in range(0,len(dataset_trimmed)):
    f = 0
    f = f_anova(i)
    if(math.isnan(f)):
        f=float('Inf')
    fscore_list.insert(i,f)



fscore_list_sorted = []
fscore_list_sorted = sorted(fscore_list,reverse= True)
fscore_list_sorted_idx = sorted(range(len(fscore_list)), key=lambda k: fscore_list[k],reverse =True)
print("#################### FEATURE SELECTION ########################")
print("###################### The top 100 features are: #############################")
print(fscore_list_sorted[0:100])
print("##################### The top 100 feature numbers are: #####################")
fscore_list_sorted_idx_100 = fscore_list_sorted_idx[0:100]
print(fscore_list_sorted_idx_100) 
dataset_100 = np.array([dataset_trimmed[j] for j in fscore_list_sorted_idx_100])
test_X_100 = np.array([test_X[k] for k in fscore_list_sorted_idx_100])

svmClf=svm.SVC(kernel='linear',C=1)
train_X_100 = dataset_100
train_X_100 = train_X_100.T
test_X_100 = test_X_100.T
train_Y = np.array([1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4])
svmClf.fit(train_X_100,train_Y)
svmPredict=list(svmClf.predict(list(test_X_100)))
#svmMeanAccuracy=svmClf.score(test_X,test_Y,sample_weight=None)
print("--------------------------------------- SVM CLASSIFIER----------------------------------")
print("The prediction label for SVM classifier is",svmPredict)
#print("The mean accuracy for SVM classifier is",svmMeanAccuracy)
################################################################## K NEAREST NEIGHBOURS##############################################3

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_X_100,train_Y)
predictk = neigh.predict(test_X_100)
print("--------------------------------------------K Nearest Neighbours-----------------------------------")
print("The Predict label for the given dataset is: ",list(predictk))

################################################## Nearest Centroid classifier #############################################

nearestCentriodclf = NearestCentroid()
nearestCentriodclf.fit(train_X_100,train_Y)
predict = nearestCentriodclf.predict(test_X_100)

print("---------------------------------- CENTROID CLASSIFIER-------------------------")
print("The Precict label for centroid classifier is: ",list(predict))



####################################### LINEAR REGRESSION #####################################
regr = linear_model.LinearRegression()
regr.fit(train_X_100,train_Y)
y_pred = regr.predict(test_X_100)
predict = [int(round(x)) for x in y_pred]
print("---------------------------------- LINEAR CLASSIFIER-------------------------")
print("The Precict label for Linear classifier is:   ",list(predict))


