import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# I understand your intention, but, I suggest to pass feature as input, not the globally shared one.
# For each function, this should be perferred:
"""
def RunSomeClassifier(X, y):
    for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        clf = ClassifierNameFromSklearn(C=c, other parameters...)
        average_scores = np.mean(cross_val_score(clf, X, y, cv=10))
        print("CLF-Short-Name:", "\t", c, "\t", round(average_scores,2)*100, "%")
"""
# Try to make variable names more relate to its function, e.g. X is a set of feature, that's right, but the correct meaning should be training data. 
# You should just called it X, and label as y.


def RunLogisticRegression(Label):
    for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        clf = LogisticRegression(max_iter = 5000, C = c).fit(Feature, Label)
        clf.predict(Feature[:2, :])
        clf.predict_proba(Feature[:2, :])
        average_scores = sum(cross_val_score(clf, Feature, Label, cv = 10))/10
        print("LR:", "\t", c, "\t", round(average_scores,2)*100, "%")

def RunSVM(Label):
    for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        clf = svm.SVC(kernel = "linear", C = c, random_state = 42)
        average_scores = sum(cross_val_score(clf, Feature, Label, cv = 10))/10
        print("SVM:", "\t", c, "\t", round(average_scores,4)*100, "%")

def RunDecisionTrees(Label):
    for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(Feature, Label)
        average_scores = sum(cross_val_score(clf, Feature, Label, cv = 10))/10
        print("DT:", "\t", c, "\t", round(average_scores,4)*100, "%")

def RunForestsofrandomizedtrees(Label):
    for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        clf = RandomForestClassifier(n_estimators = 10)
        clf = clf.fit(Feature, Label)
        average_scores = sum(cross_val_score(clf, Feature, Label, cv = 10))/10
        print("FORT:", "\t", c, "\t", round(average_scores,4)*100, "%")

Sheetsnames = pd.ExcelFile("main_data.xlsx").sheet_names

data = pd.read_excel('main_data.xlsx', sheet_name = 1, skiprows = lambda x: x in [0], index_col=0)  
test = data.to_numpy()
Feature = np.array([i[1:19] for i in test])

# train, test sets are really need to name carefully.
# I'll rename test as train_test, cause, this is the test for training
# In the community, people usually use test to represent real test data, not the one from training

for Labels in range(19, 26):
    print(data.keys()[Labels], "\t")
    Label = [i[Labels] for i in test]
    RunLogisticRegression(Label)
    RunSVM(Label)
    RunDecisionTrees(Label)
    RunForestsofrandomizedtrees(Label)
