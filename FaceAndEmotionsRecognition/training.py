from imutils import face_utils
import dlib
import cv2

allEmotions = []
allEmotions.append("anger")
allEmotions.append("happiness")
allEmotions.append("disgust")
allEmotions.append("surprise")
allEmotions.append("sadness")
allEmotions.append("neutral")
allEmotions.append("contempt")
allValues = {}
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
def findAllPoints(image):
    detections = detector(image)
    for _, rect in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, rect)  # Draw Facial Landmarks with the predictor class
        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        allPointsAndDistances = []
        for x, y in shape:
            allPointsAndDistances.append(x)
            allPointsAndDistances.append(y)
        if len(detections)>0:
            allValues['values'] = allPointsAndDistances
        else:
            allValues['values'] = -5000
def getAllValues(path, lab, findInPath, findInLabel):
    training, labels = [], []
    pathTrain, patValues = [], []
    for emotion in allEmotions:
        for a, b in zip(findInPath, findInLabel):
            pathTrain.append(a)
            patValues.append(b)
        for item, name in zip(pathTrain, patValues):
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            findAllPoints(gray)
            if allValues["values"] != -5000:
                training.append(allValues['values'])  # append image array to training data list
                labels.append(name)
    return training, labels