#!/usr/bin/env python3

from server import server
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy
import time
import collections

center = 0.5
gesture_buffer = collections.deque(maxlen=200)
thres_buffer = collections.deque(maxlen=200)
gesture_recording_idx = -1
classifier = None

class Classifier:
    """ Simple KNN classifier trained with what's in 'train', evaluated with
        what's in 'test'
    """
    trained_gestures = {
        -1: 'No gesture',
        0: 'Left and Stay',
        1: 'Left and Back',
        2: 'Right and Stay',
        3: 'Right and Back',
    }

    training_files = {
        0: ['train/leftStay.npy', 'train/leftStay2.npy', 'train/TuetuopayLeftStay.npy'],
        1: ['train/leftBack.npy', 'train/TuetuopayLeftBack.npy'],
        2: ['train/rightStay.npy', 'train/rightStay2.npy', 'train/TuetuopayRightStay.npy'],
        3: ['train/rightBack.npy', 'train/TuetuopayRightBack.npy']
    }

    validation_files = {
        0: ['test/testLeftStay.npy'],
        1: ['test/testLeftBack.npy'],
        2: ['test/testRightStay.npy', 'train/mattRightStay.npy'],
        3: ['test/testRightBack.npy']
    }

    def __predict_to_corr__(self, prediction, thres=0.5):
        """ Simple converter utility to convert from KNN output to gesture index
            If the KNN detected no gesture, or more than one gesture with a
            probability of more than thres, the function will return -1
        """
        corrs = []
        for i in range(len(prediction)):
            if prediction[i] > thres:
                corrs.append(i)
        if len(corrs) != 1:
            return -1
        else:
            return corrs[0]

    def __append_dataset__(self, X, Y, Yidx, fn, isFirst=False):
        """ Utility function to append a specific dataset named fn to X and Y
            The Yidx is the index in Y where the 1 has to be put on usually the
            gesture index
        """
        tmpX = numpy.load(fn)
        # Remove one to remove the 'No Gesture' gesture :)
        tmpY = numpy.ones(len(self.trained_gestures) - 1)*0
        tmpY[Yidx] = 1

        if isFirst:
            X = tmpX
        else:
            X = numpy.concatenate((X, tmpX))
        for i in range(len(tmpX)):
            Y.append(tmpY.tolist())

        return (X, Y)

    def __import_dataset__(self, dataset):
        """ Import the dataset and return it with its labels """
        X = []
        Y = []
        firstIter = True

        for gesture, files in dataset.items():
            for cur_file in files:
                X, Y = self.__append_dataset__(X, Y,
                        gesture, cur_file, isFirst=firstIter)
                if firstIter:
                    firstIter = False
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

        return (X, Y)

    def __init__(self):
        self.X_train, self.Y_train = self.__import_dataset__(self.training_files)
        self.X_test, self.Y_test = self.__import_dataset__(self.validation_files)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(self.X_train, self.Y_train)

    def evaluate_knn(self):
        """ Evaluate the KNN with validation data set and return accuracy of the KNN
        """
        predictions = self.knn.predict(self.X_test)
        nb_err = 0
        for i in range(len(predictions)):
            if numpy.sum(abs(predictions[i] - self.Y_test[i])) > 0.1:
                print("ERROR on " + str(i))
                print("Predicted " + self.trained_gestures[self.__predict_to_corr__(predictions[i])] + \
                      " but it was " + self.trained_gestures[self.__predict_to_corr__(self.Y_test[i])])
                nb_err += 1
        acc = 1 - nb_err/len(predictions)
        print("Evaluation results: " + str(100*acc) + "% accuracy")
        return acc

    def classify_gesture(self, gesture):
        """ Categorize a gesture and return its ID, or -1 if it failed to detect any gesture
        """
        prediction = self.knn.predict(numpy.reshape(gesture, (1, -1)))[0]
        prediction_idx = self.__predict_to_corr__(prediction)
        print("Recognized " + self.trained_gestures[prediction_idx])
        return prediction_idx

def dump(ax, ay, az, gx, gy, gz):
    global gesture_buffer, gesture_recording_idx

    # Save to buffer
    gesture_buffer.append([ax, ay, az, gx, gy, gz])
    thres_buffer.append((ax-center)**2 + (ay-center)**2 + (az-center)**2)

    # TODO: Calibrate this value
    if len(thres_buffer) > 20 and numpy.sum(list(thres_buffer)[-20:-1]) > 0.1 and gesture_recording_idx == -1:
        gesture_recording_idx = 0

    if gesture_recording_idx != -1:
        gesture_recording_idx += 1
        # To keep n points before recognition, substract them to the maxlen of queue
        if gesture_recording_idx == 180:
            # Gesture is recorded, stop now
            gesture_recording_idx = -1
            # And send to classifier
            classifier.classify_gesture([i for axis in list(gesture_buffer) for i in axis])

def helper():
    print("Press enter to finish")
    time.sleep(1)

def main():
    global classifier
    # Prepare the classifier
    classifier = Classifier()

    # Check it
    classifier.evaluate_knn()

    # Instanciate a UDP server
    server.start(callback = dump)

if __name__ == '__main__':
    helper();
    main()
    input()

