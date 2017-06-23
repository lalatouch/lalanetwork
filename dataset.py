#!/usr/bin/env python3

from server import server
import numpy

total_dumps = []
cur_gesture = []
recording = False

def dump(ax, ay, az, gx, gy, gz):
    global total_dumps, cur_gesture, recording

    if not recording:
        print("Please press enter when ready to record")
        return

    total_dumps.append([ax, ay, az, gx, gy, gz])
    print("A = ({}, {}, {}), G = ({}, {}, {})".format(ax, ay, az, gx, gy, gz))

    if(len(total_dumps) == 200):
        cur_gesture.append(total_dumps)
        total_dumps = []
        recording = False

    if(len(cur_gesture) == 15):
        numpy.save("train.csv", numpy.array(cur_gesture))
        cur_gesture = []
        print("saved training set")
        

server.start(callback = dump)
for i in range(16):
    toto = input()
    recording = True


