#%%
# Imports
from sklearn import tree
from sklearn import metrics
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../tools/')
sys.path.insert(1,path)
from email_preprocess import preprocess
from file_op import dos2unix

# %% 
dos2unix("../tools/email_authors.pkl","../tools/email_authors_fixed.pkl")
dos2unix("../tools/word_data.pkl","../tools/word_data_fixed.pkl")

features_train, features_test, labels_train, labels_test = preprocess()
# %%
# Train and predict
clf = tree.DecisionTreeClassifier()
pred = clf.fit(features_train, labels_train).predict(features_test)
print("Accuracy:",metrics.accuracy_score(labels_test, pred))

# %%
