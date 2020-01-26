from imutils import face_utils
import numpy as np
import dlib
import os
import cv2
import joblib
from sklearn.svm import SVC
import training
clf = SVC(kernel='linear')
tl = ''
tt = ''
def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    tl = train_image_labels
    tt = train_image_paths
    if os.path.isfile('./classifierZad3.pkl'):
        tl = train_image_labels
        tt = train_image_paths
        return joblib.load('./classifierZad3.pkl')
    else:
        tl = train_image_labels
        tt = train_image_paths
        inputs = ''
        outputs = ''
        result = training.getAllValues(train_image_paths, train_image_labels, train_image_paths, train_image_labels)
        train, labels = result
        inputs = np.array(train)  # Turn the training set into a numpy array for the classifier
        outputs = np.array(labels)
        clf.fit(inputs, outputs)
        joblib.dump(clf, "./classifierZad3.pkl", compress=9)
        model = clf
        return model


def extract_facial_expression_from_image(trained_model, image_path):
    image = cv2.imread(image_path)  # open image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    training.findAllPoints(gray)
    array = []
    array.append(training.allValues['values'])
    clf = trained_model
    stri = str(clf.predict(array)).replace("['", "").replace("']", "")
    facial_expression = stri
    return facial_expression
