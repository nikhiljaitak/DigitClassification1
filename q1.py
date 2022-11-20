import matplotlib.pyplot as plt

import pytest

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split



RANDOM_SEED = 42

def train():

    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False, random_state=RANDOM_SEED
)

# Learn the digits on the train subset
    clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
    predictions = clf.predict(X_test)

    return ( predictions, y_test, clf )

def accuracy_metric(y_test, predicted, clf):


    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    print(' USING RANDOM STATE 42 AND SHUFFLE FALSE WE CAN GET THE SAME SAMPLES WHILE SPLITTING')


def test_random_seed_same():
    assert RANDOM_SEED == 42

def test_random_seed_diff():
    assert RANDOM_SEED != 42

predictions, y_test, clf = train()
accuracy_metric(predictions, y_test, clf)

