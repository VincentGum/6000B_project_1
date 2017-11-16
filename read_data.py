import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

train_data_file = pd.read_csv('data/traindata.csv')
train_labels_file = pd.read_csv('data/trainlabel.csv')
test_data_file = pd.read_csv('data/testdata.csv')

for i in train_data_file:
    train_data_file[i] = train_data_file[i] - np.mean(train_data_file[i])

attributes = train_data_file.values
labels = train_labels_file.values.ravel()
X_test = test_data_file.values

accuracy_mean = []
accuracy_ = []

log_cls = LogisticRegression()
dtree_clf = DecisionTreeClassifier(random_state=0)
gnb_clf = GaussianNB()
rf_clf = RandomForestClassifier(max_depth=2, random_state=0)

kf = KFold(n_splits=5, shuffle=False, random_state=None)
kf.get_n_splits(attributes)
for train_index, test_index in kf.split(attributes):
    X_train, X_test = attributes[train_index], attributes[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # print(type(X_train))
    # print(X_train.shape)

    accuracy = []

    # using svm
    # svm_cls = svm.SVC(kernel='linear')
    # svm_cls.fit(X_train, y_train)
    # svm_predict = svm_cls.score(X_test, y_test)
    # accuracy.append(svm_predict)

    # using logistic regression
    log_cls.fit(X_train, y_train)
    log_accuracy = log_cls.score(X_test, y_test)
    accuracy.append(log_accuracy)

    # using decision tree
    dtree_clf.fit(X_train, y_train)
    dtree_accuracy = dtree_clf.score(X_test, y_test)
    accuracy.append(dtree_accuracy)

    # using gaussian naive bayes
    gnb_clf.fit(X_train, y_train)
    gnb_accuracy = gnb_clf.score(X_test, y_test)
    accuracy.append(gnb_accuracy)

    # using random forest
    rf_clf.fit(X_train, y_train)
    rf_accuracy = rf_clf.score(X_test, y_test)
    accuracy.append(rf_accuracy)

    accuracy_.append(accuracy)

for i in range(4):
    accuracy_mean.append((accuracy_[0][i] + accuracy_[1][i] + accuracy_[2][i] + accuracy_[3][i])/4)

print(accuracy_mean)

# as the first classifier gives the best performance in cross validation,
# therefore, it is chosen.

result = log_cls.predict(test_data_file)
result = pd.Series(result)
result.to_csv('project1_20446377.csv',index=False)




