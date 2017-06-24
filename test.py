#!/usr/bin/env python3

from server import server
import numpy
from keras.models import load_model

total_dumps = []
cur_gesture = []
recording = False

model = load_model('testLeftRight.h5')

def dump(ax, ay, az, gx, gy, gz):
    global total_dumps, cur_gesture, recording

    if not recording:
        #print("Please press enter when ready to record")
        return

    total_dumps.append([ax, ay, az, gx, gy, gz])
    print("A = ({}, {}, {}), G = ({}, {}, {})".format(ax, ay, az, gx, gy, gz))

    if(len(total_dumps) == 200):
        print(model.predict([total_dumps], batch_size=128))
        total_dumps = []
        recording = False


server.start(callback = dump)
for i in range(101):
    toto = input()
    recording = True


