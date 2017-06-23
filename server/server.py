#!/bin/env python3
# -*- encoding: utf8 -*-

import socket, threading, struct

# Global definitions
UDP_IP = "0.0.0.0"

sock = None
thread_sock = None

##
# Starts the UDP server
# @param port
#
def start(port = 1414):
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, port))

    thread_sock = threading.Thread(target=listen_thread)
    thread_sock.daemon = True
    thread_sock.start()

def listen_thread():
    global sock

    while True:
        data, addr = sock.recvfrom(2 * 6) # 6 values of 2 bytes
        print("Received message: ", data)
        # Unpack data
        ax, ay, az, gx, gy, gz = struct.unpack("=hhhhhh", data)
        print("A = ({}, {}, {}), G = ({}, {}, {})".format(ax, ay, az, gx, gy, gz))

