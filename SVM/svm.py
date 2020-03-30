
#%%
from sklearn import svm
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../tools/')
sys.path.insert(1,path)
from email_preprocess import preprocess
from sklearn import metrics
import time

#%%
# split data
features_train, features_test, labels_train, labels_test = preprocess()
# one persent of the train data
features_train_1 = features_train[:int(len(features_train)/100)]
labels_train_1 = labels_train[:int(len(labels_train)/100)]
#%%
# linear kernel
clf = svm.SVC(kernel='linear')
start_time = time.time()
pred = clf.fit(features_train, labels_train).predict(features_test)
print("Accuracy:",metrics.accuracy_score(labels_test, pred))
print(round(time.time()-start_time,2)," seconds")

# %%
# Exhaustive Grid Search
c_values = [0.001,0.01,0.1,1,10,100,500,1000,5000]
gamma_values = [0.001,0.01,0.1,10,20,100,500]
kernels = ['linear', 'rbf', 'sigmoid','poly']
results = []
# test with 1% of the data for performance puposes
for c in c_values:
    for k in kernels:
        if k != 'rbf':
            start_time = time.time()
            clf = svm.SVC(kernel=k,C=c)
            pred = clf.fit(features_train_1, labels_train_1).predict(features_test)
            duration = time.time()-start_time
            acc = metrics.accuracy_score(labels_test, pred)
            print('kernel:',k,"C:",c)
            print("Accuracy:",acc)
            print(round(time.time()-start_time,2)," seconds \n")
            results.append({'kernel':k, 'C':c,'Accuracy':acc,'time':duration})
        else:
            for g in gamma_values:
                start_time = time.time()
                clf = svm.SVC(kernel=k,C=c, gamma=g)
                pred = clf.fit(features_train_1, labels_train_1).predict(features_test)
                duration = time.time()-start_time
                acc = metrics.accuracy_score(labels_test, pred)
                print('kernel:',k,"C:",c, 'gamma:',g)
                print("Accuracy:",acc)
                print(round(time.time()-start_time,2)," seconds \n")
                results.append({'kernel':k, 'C':c, 'gamma':g,'Accuracy':acc,'time':duration})

#%%
best = max(results, key=lambda x: x['Accuracy'])
best_kernel = best['kernel']
best_c = best['C']
best_gamma = best['gamma']

print(best_kernel, best_c, best['Accuracy'])

# %%
# rbf kernel with the best parameters
clf = svm.SVC(kernel=best_kernel, C = best_c, gamma = best_gamma)
pred = clf.fit(features_train, labels_train).predict(features_test)
print("Accuracy:",metrics.accuracy_score(labels_test, pred))
print(round(time.time()-start_time,2)," seconds")


# %%
# Parameter estimation using grid search with cross-validation
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf','sigmoid','poly'),
                'C':[0.001,0.01,0.1,1,10,100,500,1000,5000],
                 'gamma':[0.001,0.01,0.1,10,20,100,500]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
pred = clf.fit(features_train_1, labels_train_1).predict(features_test)
print("Accuracy:",metrics.accuracy_score(labels_test, pred))


# %%
