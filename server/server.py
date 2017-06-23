#!/bin/env python3
# -*- encoding: utf8 -*-

import socket

# Global definitions
UDP_IP = "0.0.0.0"

sock = None

##
# Starts the UDP server
# @param port
#
def start(port = 1414):
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, port))

def listen_thread():
    global sock

    while True:
        data, addr = sock.recvfrom(1024)
        print("Received message: ", data)

