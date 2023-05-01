import calculaterobotposition
import threading
import socket
import struct
import time


#IP and port to send data to
localIP = ""
localPort = 3333
bufferSize  = 1024

#Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

#Bind to address and ip
UDPServerSocket.bind((localIP, localPort))

print("UDP server up and listening")


#Checks connection from robot to server
bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
print("Mouse Address Received")

message = bytesAddressPair[0]
address = bytesAddressPair[1]

clientMsg = "Message from Client:{}".format(message)
clientIP  = "Client IP Address:{}".format(address)
    
print(clientMsg)
print(clientIP)


x = calculaterobotposition.Coordinates()
positionthread = threading.Thread(target = x.calculaterobotposition)
positionthread.start()

while(True):
   

   #Get phi variable from calculaterobotposition.py
   phi = x.phi
   
   #Send data to mouse
   offset_pack = struct.pack("f", phi)
   UDPServerSocket.sendto(offset_pack, address)
   
   time.sleep(0.2)



