#!/bin/env python3
# -*- encoding: utf8 -*-

import socket, time

# Global definitions
UDP_IP = "255.255.255.255"
UDP_PORT = 1415

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

while True:
    sock.sendto(bytes([42]), (UDP_IP, UDP_PORT))
    time.sleep(1)

