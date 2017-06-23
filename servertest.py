#!/usr/bin/env python3

from server import server

def dump(ax, ay, az, gx, gy, gz):
    print("A = ({}, {}, {}), G = ({}, {}, {})".format(ax, ay, az, gx, gy, gz))

server.start(callback = dump)
toto = input()

