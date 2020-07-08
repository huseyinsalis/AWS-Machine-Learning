
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import time
sys.path.append("..\\tools\\")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]



#########################################################
### your code goes here ###
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', C=10000)
t1= time.time()
classifier.fit(features_train, labels_train)
print(time.time()-t1)
y_pred = classifier.predict(features_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, y_pred)
tn, fp, fn, tp = confusion_matrix(labels_test, y_pred).ravel()
print(cm)
print("Traininng Accuracy:{:.2f}".format(classifier.score(features_train, labels_train)))
print("Test Accuracy:{:.2f}".format(classifier.score(features_test, labels_test)))
precision=tp/(tp+fp)
print("Precision:{:.2f}".format(precision))
recall=tp/(tp+fn)
F1_score=2*precision*recall/(precision+recall)
print("Recall:{:.2f}".format(precision))
print("F1 Score:{:.2f}".format(precision))
#########################################################