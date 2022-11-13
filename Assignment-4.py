#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Standard scientific Python imports
from unittest.mock import DEFAULT
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


GAMMAS = [0.005,0.001,0.05,0.01,0.1,0.5]
Clist=[0.1,0.5,1,3,5]
DEFAULT_C=0.5
DEFAULT_GAMMA=0.005

MAX_DEPTH= [2, 3, 5, 10, 20]
MIN_SAMPLES_LEAF= [5, 10, 20, 50, 100]
CRITERION= ["gini", "entropy"]

DEFAULT_MAX_DEPTH=2
DEFAULT_SAMPLES_LEAF = 5
DEFAULT_CRITERION= "gini"

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
def reshapeImages(inputDigits):
    n_samples = len(inputDigits.images)
    reshapedImages = inputDigits.images.reshape((n_samples, -1))
    return reshapedImages


#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model

def decisiontree(X_train, y_train, X_test, y_test):
     from sklearn import tree
     clf = tree.DecisionTreeClassifier()  
     clf.fit(X_train, y_train)
     predicted_test = clf.predict(X_test)
     test_accuracy = metrics.accuracy_score(predicted_test,y_test)
     return test_accuracy

def SVM(X_train, y_train, X_dev, y_dev, X_test, y_test, best_test_Accuracy, best_dev_Accuracy, best_hyper_params, gamma, c):
                                             
    #PART: Define the model
# Create a classifier: a support vector classifier
    clf = svm.SVC()                                           

#PART: setting up hyperparameter
    hyper_params = {'gamma':gamma,'C':c}
    clf.set_params(**hyper_params)


#PART: Train model
# Learn the digits on the train subset
    clf.fit(X_train, y_train)

#PART: Get test set predictions
# Predict the value of the digit on the test subset
    predicted_test = clf.predict(X_test)
    predicted_dev = clf.predict(X_dev)

#PART: Compute evaluation metrics on dev and test
    dev_Accuracy = metrics.accuracy_score(predicted_dev, y_dev )
    test_accuracy = metrics.accuracy_score(predicted_test,y_test)

    #print('dev accuracy: ',metrics.accuracy_score(predicted_dev, y_dev ))
    #print('test accuracy: ',metrics.accuracy_score(predicted_test,y_test))

    if dev_Accuracy > best_dev_Accuracy:
        return test_accuracy, dev_Accuracy, hyper_params
    else:
        return best_test_Accuracy, best_dev_Accuracy, best_hyper_params

def Decision_Tree(X_train, y_train, X_dev, y_dev, X_test, y_test, best_test_Accuracy, best_dev_Accuracy, best_hyper_params, depth, min_samples, criterion):
                                             
    #PART: Define the model
# Create a classifier: a support vector classifier
    #clf = svm.SVC()                                           
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()  
#PART: setting up hyperparameter
    hyper_params = { 'max_depth': depth,'min_samples_leaf':min_samples,'criterion': criterion}
    clf.set_params(**hyper_params)


#PART: Train model
# Learn the digits on the train subset
    clf.fit(X_train, y_train)

#PART: Get test set predictions
# Predict the value of the digit on the test subset
    predicted_test = clf.predict(X_test)
    predicted_dev = clf.predict(X_dev)

#PART: Compute evaluation metrics on dev and test
    dev_Accuracy = metrics.accuracy_score(predicted_dev, y_dev )
    test_accuracy = metrics.accuracy_score(predicted_test,y_test)

    #print('dev accuracy: ',metrics.accuracy_score(predicted_dev, y_dev ))
    #print('test accuracy: ',metrics.accuracy_score(predicted_test,y_test))

    if dev_Accuracy > best_dev_Accuracy:
        return test_accuracy, dev_Accuracy, hyper_params
    else:
        return best_test_Accuracy, best_dev_Accuracy, best_hyper_params

import pandas as pd
df=pd.DataFrame(columns=['gamma','c'])
reshapedImages=reshapeImages(digits)
index=0
for gamma in GAMMAS:
    for c in Clist:
        df.at[index,'gamma']=gamma
        df.at[index,'c']=c
        index+=1
index=0
dfHyperDecision = pd.DataFrame(columns=['max_depth','min_samples','criterion'])
for criteria in CRITERION:
    for depth in MAX_DEPTH:
        for sample in MIN_SAMPLES_LEAF:
            dfHyperDecision.at[index,'max_depth']=depth
            dfHyperDecision.at[index,'min_samples']=sample
            dfHyperDecision.at[index,'criterion']=criteria
            index+=1

print('total hyperparamters Decision Tree',len(dfHyperDecision))
print('total hyperparamters SVM',len(df))

from sklearn.model_selection import KFold
X=reshapedImages
y=digits.target
kfold = KFold(5)
foldNum=-1

dfFinal=pd.DataFrame(columns=['RUN','SVM','DECISION_TREE','SVM_BEST','DECISION_BEST'])

for train_index, test_index in kfold.split(reshapedImages):
    
    best_dev_Accuracy,tree_best_dev_Accuracy = 0,0
    best_test_Accuracy, tree_best_test_Accuracy = 0,0

    best_hyper_params = {'gamma':DEFAULT_GAMMA,'C':DEFAULT_C}
    tree_best_hyper_params={ 'max_depth': DEFAULT_MAX_DEPTH,'min_samples_leaf':DEFAULT_SAMPLES_LEAF,'criterion': DEFAULT_CRITERION}
    foldNum+=1

    dev_test_frac = 1-train_frac

    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        reshapedImages, digits.target, test_size=dev_test_frac, shuffle=True
                                                                  )
    X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True)

    
    for index,row in df.iterrows():
        best_test_Accuracy, best_dev_Accuracy, best_hyper_params = SVM(X_train, y_train, X_dev, y_dev, X_test, y_test, best_test_Accuracy, best_dev_Accuracy,best_hyper_params,row['gamma'],row['c'])
        print(foldNum, best_test_Accuracy, best_dev_Accuracy, best_hyper_params)
    

    for index,row in dfHyperDecision.iterrows():                                                                 
        tree_best_test_Accuracy, tree_best_dev_Accuracy, tree_best_hyper_params = Decision_Tree(X_train, y_train, X_dev, y_dev, X_test, y_test, tree_best_test_Accuracy, tree_best_dev_Accuracy,tree_best_hyper_params,row['max_depth'],row['min_samples'], row['criterion'])
        print(foldNum, tree_best_test_Accuracy, tree_best_dev_Accuracy, tree_best_hyper_params)
    
    dfFinal.at[foldNum,'RUN']=foldNum
    dfFinal.at[foldNum,'SVM']=best_test_Accuracy
    dfFinal.at[foldNum,'DECISION_TREE']=tree_best_test_Accuracy
    dfFinal.at[foldNum,'SVM_BEST']=str(best_hyper_params)
    dfFinal.at[foldNum,'DECISION_BEST']=str(tree_best_hyper_params)

print(dfFinal.head())

print('mean svm',dfFinal['SVM'].mean())
print('mean decision',dfFinal['DECISION_TREE'].mean())

print('std svm',dfFinal['SVM'].std())
print('std decision',dfFinal['DECISION_TREE'].std())


print('digits shape of pixels:', digits.images.shape)

