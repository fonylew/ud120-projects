#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



##############################
### your code goes here ###
# smaller dataset
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#########################################################
#clf = SVC(kernel='linear')
clf = SVC(C=10000 ,kernel='rbf')
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "predict  time:", round(time()-t1, 3), "s"
accuracy = accuracy_score(labels_test, pred)
print accuracy
# trained with smaller dataset
# C = 1     0.616040955631
# C = 10    0.616040955631
# C = 100;  0.616040955631
# C = 1000  0.821387940842
# C = 10000 0.892491467577

# trained with full dataset
# C = 10000 (rbf)   0.990898748578
chris = 0
for i in pred:
    if i == 1: 
        chris += 1
print "predicted christ : " , chris
(a,b,c) = clf.predict((features_test[10],features_test[26],features_test[50]))
print (a,b,c)
print a == labels_test[10]
print b == labels_test[26]
print c == labels_test[50]