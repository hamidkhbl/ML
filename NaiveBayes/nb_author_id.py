""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
#%%
# imports 
import sys
import os
from time import time
import pickle
import numpy
path = os.path.join(os.path.dirname(__file__), '../tools/')
sys.path.insert(1,path)
from email_preprocess import preprocess
from file_op import dos2unix

#%%
# Fix files, only run it once if the files are not in a correct formst
dos2unix("../tools/email_authors.pkl","../tools/email_authors_fixed.pkl")
dos2unix("../tools/word_data.pkl","../tools/word_data_fixed.pkl")

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#%%

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
clf = GaussianNB()
pred = clf.fit(features_train, labels_train).predict(features_test)
print("Accuracy:",metrics.accuracy_score(labels_test, pred))
#########################################################
# %%
