import MicStreaming
import calculaterobotposition
import robotwifi
import time
import threading
import socket
import struct

# IP and port to send data to
localIP = ""
localPort = 3333
bufferSize = 1024

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))

print("UDP server up and listening")

# Checks connection from robot to server
bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
print("Mouse Address Received")

message = bytesAddressPair[0]
address = bytesAddressPair[1]

clientMsg = "Message from Client:{}".format(message)
clientIP = "Client IP Address:{}".format(address)

print(clientMsg)
print(clientIP)

# Initialize Camera and Microphone

# Initialize Microphone
mic1 = MicStreaming.Streaming()
mic1.device_index = 1
mic1.start_recording()
thread1 = threading.Thread(target=mic1.stream_read)
thread1.start()
time.sleep(1)
mic1.magnitude()

# Initialize Camera
x = calculaterobotposition.Coordinates()
positionthread = threading.Thread(target=x.calculaterobotposition)
positionthread.start()

while (True):
    # Get phi variable from calculaterobotposition.py
    phi = x.phi
    distance = mic1.distance

    # Send data to mouse
    offset_pack = struct.pack("f", phi, distance)
    UDPServerSocket.sendto(offset_pack, address)

    time.sleep(0.2)





