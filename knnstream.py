#!/usr/bin/env python3

from server import server
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy
import time
import collections
from urllib.request import urlopen
import os

# Change this if you want to record data
record_new_dataset = False
recording = 0
record_dataset = []

started = False
center = [0.5]*6
rollback = 50
threshold = 0.15
gesture_buffer = collections.deque(maxlen=200)
gesture_recording_idx = -600
classifier = None
calibrating = True
to_plot = []

G_LEFT_AND_STAY, G_LEFT_AND_BACK, G_RIGHT_AND_STAY, G_RIGHT_AND_BACK = 0, 1, 2, 3

API_URL = os.environ.get("API_URL", None)
if API_URL is None: API_URL = "localhost"
API_PORT = os.environ.get("API_PORT", None)
if API_PORT is None: API_PORT = 3000

print("API is at {}:{}".format(API_URL, API_PORT))

def api_get(url):
    return urlopen("http://{}:{}/api{}".format(API_URL, API_PORT, url))

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
        0: ['train/0-didjcodt.npy', 'train/0-didjcodt2.npy', 'train/0-tuetuopay.npy'],
        1: ['train/1-didjcodt.npy', 'train/1-didjcodt2.npy', 'train/1-tuetuopay.npy'],
        2: ['train/2-didjcodt.npy', 'train/2-didjcodt2.npy', 'train/2-tuetuopay.npy'],
        3: ['train/3-didjcodt.npy', 'train/3-didjcodt2.npy', 'train/3-tuetuopay.npy']
    }

    validation_files = {
        0: ['test/0-didjcodt.npy'],
        1: ['test/1-didjcodt.npy'],
        2: ['test/2-didjcodt.npy'],
        3: ['test/3-didjcodt.npy']
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
        shaped_X = []
        for sample in tmpX:
             shaped_X.append(numpy.concatenate((sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3], sample[:, 4], sample[:, 5])))
        # Remove one to remove the 'No Gesture' gesture :)
        tmpY = numpy.ones(len(self.trained_gestures) - 1)*0
        tmpY[Yidx] = 1

        if isFirst:
            X = shaped_X
        else:
            X = numpy.concatenate((X, shaped_X))
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

        return (X, Y)

    def __init__(self):
        self.X_train, self.Y_train = self.__import_dataset__(self.training_files)
        self.X_test, self.Y_test = self.__import_dataset__(self.validation_files)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(self.X_train, self.Y_train)

        # For API client
        self.fast_moving = False

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
        print(prediction)
        print("Recognized " + self.trained_gestures[prediction_idx])

        if prediction_idx == G_RIGHT_AND_BACK:
            # TODO : next song
            print("TODO")
        elif prediction_idx == G_LEFT_AND_BACK:
            # TODO : previous song
            print("TODO")
        elif prediction_idx == G_RIGHT_AND_STAY:
            # Differentiate between fast-forward and stop backwards
            if self.fast_moving:
                # Stop going backwards
                api_get("/curTracks/fast/backward/stop")
            else:
                # Fast forward
                api_get("/curTracks/fast/forward/go")
            self.fast_moving = not self.fast_moving
        elif prediction_idx == G_LEFT_AND_STAY:
            # Differentiate between backward and stop forward
            if self.fast_moving:
                # Stop going forward
                api_get("/curTracks/fast/forward/stop")
            else:
                # Fast backward
                api_get("/curTracks/fast/backward/go")
            self.fast_moving = not self.fast_moving
        return prediction_idx

def dump(ax, ay, az, gx, gy, gz):
    global gesture_buffer, gesture_recording_idx, recording, record_dataset, to_plot

    # Save to buffer

    if calibrating:
        gesture_buffer.append([ax, ay, az, gx, gy, gz])
        return

    gesture_buffer.append([ax - center[0], ay - center[1], az - center[2], gx - center[3], gy - center[4], gz - center[5]])

    if record_new_dataset:
        if recording < 0:
            print("Record start in " + str(recording*1./200) + "s")
            recording += 1
            return
        print("A = ({}, {}, {}), G = ({}, {}, {})".format(ax, ay, az, gx, gy, gz))
        recording += 1

        if recording == 200:
            record_dataset.append(list(gesture_buffer))
            numpy.save("training", numpy.array(record_dataset))
            recording = -600
    elif started:
        # TODO: Calibrate this value
        # Start recording conditions :
        # - Not already recording
        # - rollback last samples are all outside of [0.45, 0.55] (2nd if)
        if gesture_recording_idx == -1:
            np_buffer_window = numpy.array(list(gesture_buffer)[-rollback:-1])
            np_buffer_window_outside_up = np_buffer_window > threshold
            np_buffer_window_outside_down = np_buffer_window < -threshold
            if numpy.any(np_buffer_window_outside_up + np_buffer_window_outside_down):
                gesture_recording_idx = 0

        elif gesture_recording_idx >= 0:
            gesture_recording_idx += 1
            # To keep n points before recognition, substract them to the maxlen of queue
            if gesture_recording_idx == 200 - rollback:

                # Send to classifier
                recorded_gesture = numpy.array(list(gesture_buffer))
                to_plot = [recorded_gesture[:, 0], recorded_gesture[:, 1], recorded_gesture[:, 2], recorded_gesture[:, 3], recorded_gesture[:, 4], recorded_gesture[:, 5]]
                classifier.classify_gesture(numpy.concatenate((to_plot[0], to_plot[1], to_plot[2], to_plot[3], to_plot[4], to_plot[5])))
                # Gesture is recorded, stop and wait for at least 100 samples before recognizing other gesture
                gesture_recording_idx = -100

        else:
            gesture_recording_idx += 1

def helper():
    print("Press enter to finish")
    time.sleep(0.1)

def calibrate():
    global center
    while(len(gesture_buffer) < 200):
        time.sleep(0.1)
    center = [numpy.mean(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 0])), \
              numpy.mean(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 1])), \
              numpy.mean(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 2])), \
              numpy.mean(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 3])), \
              numpy.mean(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 4])), \
              numpy.mean(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 5]))]

def main():
    global classifier, calibrating, started
    if not record_new_dataset:
        # Prepare the classifier
        classifier = Classifier()

        # Check it
        classifier.evaluate_knn()


    # Instanciate a UDP server
    server.start(callback = dump)

    # Get zero value
    print("Press enter to start calibration, hit <C-c> when finished")
    input()
    calibrate()
    calibrating = False
    # Flush the buffer
    time.sleep(1)
    started = True

    plt.ion()
    while True:
        plt.pause(0.1)
        plt.clf()
        plt.axis([0, 200, -1, 1])
        #plt.plot(list(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 0:-1:6])))
        #plt.plot(list(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 1:-1:6])))
        #plt.plot(list(numpy.ndarray.flatten(numpy.array(gesture_buffer)[:, 2:-1:6])))
        if len(to_plot) > 0:
            plt.plot(to_plot[0])
            plt.plot(to_plot[1])
            plt.plot(to_plot[2])

        #plt.show()

    # Fake cb
    #test_dataset = classifier.X_test
    #print(len(test_dataset))

    #for s in range(len(test_dataset)):
    #    for i in range(0, len(test_dataset[0])-10, 6):
    #        dump(*test_dataset[s][i:i+6])
    #    print("Should be: " + classifier.trained_gestures[classifier.__predict_to_corr__(classifier.Y_test[s])])
    #    time.sleep(1)

if __name__ == '__main__':
    helper();
    main()
    input()

