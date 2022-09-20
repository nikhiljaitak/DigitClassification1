#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


GAMMAS = [0.005,0.001,0.05,0.01,0.1,0.5]
Clist=[0.1,0.5,1,3,5]
C=0.5
GAMMA=0.005
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
def splitting_model_performance(reshapedImagesarg, inputDigitsarg, best_dev_Accuracy, best_hyper_params, gamma, c):

    dev_test_frac = 1-train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        reshapedImagesarg, inputDigitsarg.target, test_size=dev_test_frac, shuffle=True
                                                                  )
    X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
                                                    )
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

    #PART: Sanity check of test predictions
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted_test):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    #PART: Sanity check of dev predictions
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted_dev):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

#PART: Compute evaluation metrics on dev and test
    dev_Accuracy = metrics.accuracy_score(predicted_dev, y_dev )
    test_accuracy = metrics.accuracy_score(predicted_test,y_test)

    print('dev accuracy: ',metrics.accuracy_score(predicted_dev, y_dev ))
    print('test accuracy: ',metrics.accuracy_score(predicted_test,y_test))

    if dev_Accuracy > best_dev_Accuracy:
        return dev_Accuracy, hyper_params
    else:
        return best_dev_Accuracy, best_hyper_params

import pandas as pd
df=pd.DataFrame(columns=['gamma','c'])
reshapedImages=reshapeImages(digits)
parametersList=[]
index=0
best_dev_Accuracy=0
best_hyper_params = {'gamma':GAMMA,'C':C}
for gamma in GAMMAS:
    for c in Clist:
        df.at[index,'gamma']=gamma
        df.at[index,'c']=c
        index+=1
print(len(df))

for index,row in df.iterrows():
    best_dev_Accuracy, best_hyper_params = splitting_model_performance(reshapedImages, digits,best_dev_Accuracy,best_hyper_params,row['gamma'],row['c'])
    print(best_dev_Accuracy, best_hyper_params)

print('best parameters and accuracy ', best_dev_Accuracy, best_hyper_params)
print('digits shape of pixels:',digits.images.shape)

    