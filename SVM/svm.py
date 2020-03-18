
#%%
from sklearn import svm
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../tools/')
sys.path.insert(1,path)
from email_preprocess import preprocess
from file_op import dos2unix
from sklearn import metrics
import time

#%%
# linear kernel
features_train, features_test, labels_train, labels_test = preprocess()
clf = svm.SVC(kernel='linear')
pred = clf.fit(features_train, labels_train).predict(features_test)
print("Accuracy:",metrics.accuracy_score(labels_test, pred))
print(round(time.time()-start_time,2)," seconds")

# %%
