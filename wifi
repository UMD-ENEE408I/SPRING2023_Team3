#include <Arduino.h>
#include <Adafruit_MCP3008.h>
#include <Adafruit_MPU6050.h>
#include <Encoder.h>

#include <WiFi.h>
#include <WiFiUdp.h>
#include <string.h>
#include <iostream> 

// WiFi network name and password:
const char * networkName = "408ITerps";
const char * networkPswd = "goterps2022";

//IP address to send UDP data to:
// either use the ip address of the server or 
// a network broadcast address
const char * udpAddress = "192.168.2.142";
const int udpPort = 3333;

//Are we currently connected?
boolean connected = false;

//The udp library class
WiFiUDP udp;


//wifi event handler
void WiFiEvent(WiFiEvent_t event){
    switch(event) {
      case ARDUINO_EVENT_WIFI_STA_GOT_IP:
          //When connected set 
          Serial.print("WiFi connected! IP address: ");
          Serial.println(WiFi.localIP());  
          //initializes the UDP state
          //This initializes the transfer buffer
          udp.begin(WiFi.localIP(),udpPort);
          connected = true;
          break;
      case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
          Serial.println("WiFi lost connection");
          connected = false;
          break;
      default: break;
    }
}

void connectToWiFi(const char * ssid, const char * pwd){
  Serial.println("Connecting to WiFi network: " + String(ssid));

  // delete old config
  WiFi.disconnect(true);
  //register event handler
  WiFi.onEvent(WiFiEvent);
  
  //Initiate connection
  WiFi.begin(ssid, pwd);

  Serial.println("Waiting for WIFI connection...");
}


void setup(){
  // Initilize hardware serial:
  Serial.begin(115200);
  
  // Stop the right motor by setting pin 14 low
  // this pin floats high or is pulled
  // high during the bootloader phase for some reason
  pinMode(14, OUTPUT);
  digitalWrite(14, LOW);
  delay(100);

  //Connect to the WiFi network
  connectToWiFi(networkName, networkPswd);
}

void loop(){
  //only send data when connected
  if(connected){
    //Send a packet
    udp.beginPacket(udpAddress,udpPort);
    udp.printf("Seconds since boot: %lu", millis()/1000);
    udp.endPacket();
  
  //Wait for 1 second
  delay(10);

  int packetSize = udp.parsePacket();
    if(packetSize >= 2*sizeof(float))
    {
      Serial.printf("packet size is %d\n", packetSize);
      float pos_array[1]; 
      float rot_array[1];
      udp.read((char*)pos_array, sizeof(pos_array)); 
      //udp.flush();
      udp.read((char*)rot_array, sizeof(rot_array));
      udp.flush();
      Serial.printf("received pos is %f\n", pos_array[0]);
      float pos_target = pos_array[0];
      Serial.printf("target is %f\n", pos_target);
      Serial.printf("received rot is %f\n", rot_array[0]);
      float rot_target = rot_array[0];
      Serial.printf("target is %f\n", rot_target);
    }
    else{
      Serial.printf("Nothing to print");
    }
  }
}


