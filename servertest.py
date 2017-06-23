#!/usr/bin/env python3

from server import server

def dump():
    print("A = ({}, {}, {}), G = ({}, {}, {})".format(ax, ay, az, gx, gy, gz))

server.start(callback = dump)
toto = input()

