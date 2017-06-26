#!/bin/env python3
# -*- encoding: utf8 -*-

import socket, threading, struct

# Global definitions
UDP_IP = "0.0.0.0"

sock = None
thread_sock = None
callbacks = []

##
# Starts the UDP server
# @param port
#
def start(port = 1414, callback = None):
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, port))

    thread_sock = threading.Thread(target=listen_thread)
    thread_sock.daemon = True
    thread_sock.start()

    if callback is not None:
        callbacks.append(callback)

def listen_thread():
    global sock

    while True:
        data, addr = sock.recvfrom(2 * 6) # 6 values of 2 bytes
        # Unpack data
        ax, ay, az, gx, gy, gz = struct.unpack("=hhhhhh", data)
        ax = float(ax) / float(0xFFFF) + 0.5
        ay = float(ay) / float(0xFFFF) + 0.5
        az = float(az) / float(0xFFFF) + 0.5
        gx = float(gx) / float(0xFFFF) + 0.5
        gy = float(gy) / float(0xFFFF) + 0.5
        gz = float(gz) / float(0xFFFF) + 0.5
        for cb in callbacks:
            cb(ax, ay, az, gx, gy, gz)

