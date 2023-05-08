import MicStreaming
import calculaterobotposition
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
# Set Device index
info = mic1.audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if mic1.audio.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels') > 0:
        if mic1.audio.get_device_info_by_host_api_device_index(0, i).get('name') == 'Microphone (2- USBAudio1.0)':
            mic1.device_index = i
        print('Input Device id ', i, ' - ', mic1.audio.get_device_info_by_host_api_device_index(0, i).get('name'))
print(mic1.device_index)
mic1.start_recording()
thread1 = threading.Thread(target=mic1.stream_read)
thread1.start()
time.sleep(1)
print('Mic Recording...')

Mic_thread = threading.Thread(target=mic1.magnitude)
Mic_thread.start()
print('Calculating Magnitude...')

x = calculaterobotposition.Coordinates()
positionthread = threading.Thread(target=x.calculaterobotposition)
positionthread.start()
print('Camera Started')

while (True):
    # Get phi variable from calculaterobotposition.py
    phi = x.phi
    distance = mic1.distance
    print(phi)
    if distance > 300:
        forward = False
    else:
        forward = True
    # Send data to mouse\
    offset_pack = struct.pack("ff", phi, distance)
    UDPServerSocket.sendto(offset_pack, address)






