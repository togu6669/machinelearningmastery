# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://goo.gl/vhm1eU"
fileName = "pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(fileName, names=names)
array = dataframe.values

# X = array[:,0:8]
# Y = array[:,8]

X = array[9:,0:8]
Y = array[9:,8]
Y = Y.astype('int')

model = LogisticRegression()

# k-Fold with n splits = 10
kfold = KFold(n_splits=10, random_state=7)
# Day 7
# results = cross_val_score(model, X, Y, cv=kfold)
## print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
# print("Accuracy:", results.mean()*100.0, results.std()*100.0)
# loo = LeaveOneOut() 
# results = cross_val_score(model, X, Y, cv=loo, scoring=scoring)
# print("Accuracy:", results.mean()*100.0, results.std()*100.0)

# LogLoss metrics - day 8
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
print("Logloss:", results.mean(), results.std())

# LeaveOneOut
Y = array[9:,8]
Y = Y.astype('int')
loo = LeaveOneOut() 
results = cross_val_score(model, X, Y, cv=loo)
# print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())​
print("Logloss:", results.mean(), results.std())