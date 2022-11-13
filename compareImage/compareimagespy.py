import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from flask import Flask
import os
from flask import request, jsonify
from flask_cors import CORS
import json
import os
import sys
from flask_cors import CORS
app = Flask(__name__)
CORS(app)



best_hyper_params = {'gamma': 0.001, 'C': 0.5}

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def show(inputImage):
    import matplotlib.pyplot as plt
    import matplotlib.image as img
    testImage = img.imread(inputImage)
    plt.imshow(testImage)

def train():
    import matplotlib.pyplot as plt
    from sklearn import datasets, svm, metrics
    from sklearn.model_selection import train_test_split
    digits = datasets.load_digits()
    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    # Create a classifier: a support vector classifier
    
    clf = svm.SVC()
    
    #PART: setting up hyperparameter
    clf.set_params(**best_hyper_params)
    
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
    )
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    predicted_test = clf.predict(X_test)
    
    test_accuracy = metrics.accuracy_score(predicted_test,y_test)
    
    print('Accuracy::',test_accuracy )
    #disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    #disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return clf

def predict(inputFile, clf):
    import matplotlib.image as img
    imgArr = mpimg.imread(inputFile)     
    gray = rgb2gray(imgArr) 
    print('array::', gray)
    predicted = clf.predict([gray.flatten()])
    print('predicted::',predicted)
    return predicted
    

@app.route("/api/healthcheck/", methods=["GET"])
def health():
    return jsonify({'response': 'successful'})

@app.route("/api/predict/", methods=["POST"])
def predict_images():

    args = request.args
    print(args.to_dict())
    print(args.get("image1"))
    print(args.get("image2"))
    image1=args.get("image1")
    image2=args.get("image2")
    clf = train()
    image1Pred = predict(image1, clf)
    image2Pred = predict(image2, clf)

    print(image1Pred, type(image1Pred))
    print(image2Pred, type(image2Pred))
    if str(image1Pred[0]) == str(image2Pred[0]):
        return jsonify({'response': 'same images','inputs':args})
    else:
        return jsonify({'response': 'different images'})
    #inputimagePath='8_2.png'
    #show(inputimagePath)
    #clf = train()
    #predict(inputimagePath,clf)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT'))

