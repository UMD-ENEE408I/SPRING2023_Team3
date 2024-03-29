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
const char * udpAddress = "192.168.2.126";
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



// IMU (rotation rate and acceleration)
Adafruit_MPU6050 mpu;
Adafruit_MCP3008 adc1;
Adafruit_MCP3008 adc2;

// Buzzer pin which we will use for indicating IMU initialization failure
const unsigned int BUZZ = 26;
const unsigned int BUZZ_CHANNEL = 0;

// Need these pins to turn off light bar ADC chips
const unsigned int ADC_1_CS = 2;
const unsigned int ADC_2_CS = 17;

// Battery voltage measurement constants
const unsigned int VCC_SENSE = 27;
const float ADC_COUNTS_TO_VOLTS = (2.4 + 1.0) / 1.0 * 3.3 / 4095.0;

// Motor encoder pins
const unsigned int M1_ENC_A = 39;
const unsigned int M1_ENC_B = 38;
const unsigned int M2_ENC_A = 37;
const unsigned int M2_ENC_B = 36;

// Motor power pins
const unsigned int M1_IN_1 = 13;
const unsigned int M1_IN_2 = 12;
const unsigned int M2_IN_1 = 25;
const unsigned int M2_IN_2 = 14;

// Motor PWM channels
const unsigned int M1_IN_1_CHANNEL = 8;
const unsigned int M1_IN_2_CHANNEL = 9;
const unsigned int M2_IN_1_CHANNEL = 10;
const unsigned int M2_IN_2_CHANNEL = 11;

const int M_PWM_FREQ = 5000;
const int M_PWM_BITS = 8;
const unsigned int MAX_PWM_VALUE = 255; // Max PWM given 8 bit resolution

float METERS_PER_TICK = (3.14159 * 0.031) / 360.0;
float TURNING_RADIUS_METERS = 4.3 / 100.0; // Wheels are about 4.3 cm from pivot point

void configure_imu() {
  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      ledcWriteNote(BUZZ_CHANNEL, NOTE_C, 4);
      delay(500);
      ledcWriteNote(BUZZ_CHANNEL, NOTE_G, 4);
      delay(500);
    }
  }
  Serial.println("MPU6050 Found!");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_1000_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
}

void read_imu(float& w_z) {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  w_z = g.gyro.z;
}

void est_imu_bias(float& E_w_z, int N_samples) {
  float E_w_z_acc = 0.0;
  for (unsigned int i = 0; i < N_samples; i++) {
    float w_z;
    read_imu(w_z);
    E_w_z_acc += w_z;
    delay(5);
  }
  E_w_z = E_w_z_acc / N_samples;
}

void configure_motor_pins() {
  ledcSetup(M1_IN_1_CHANNEL, M_PWM_FREQ, M_PWM_BITS);
  ledcSetup(M1_IN_2_CHANNEL, M_PWM_FREQ, M_PWM_BITS);
  ledcSetup(M2_IN_1_CHANNEL, M_PWM_FREQ, M_PWM_BITS);
  ledcSetup(M2_IN_2_CHANNEL, M_PWM_FREQ, M_PWM_BITS);

  ledcAttachPin(M1_IN_1, M1_IN_1_CHANNEL);
  ledcAttachPin(M1_IN_2, M1_IN_2_CHANNEL);
  ledcAttachPin(M2_IN_1, M2_IN_1_CHANNEL);
  ledcAttachPin(M2_IN_2, M2_IN_2_CHANNEL);
}

// Positive means forward, negative means backwards
void set_motors_pwm(float left_pwm, float right_pwm) {
  if (isnan(left_pwm)) left_pwm = 0.0;
  if (left_pwm  >  255.0) left_pwm  =  255.0;
  if (left_pwm  < -255.0) left_pwm  = -255.0;
  if (isnan(right_pwm)) right_pwm = 0.0;
  if (right_pwm >  255.0) right_pwm =  255.0;
  if (right_pwm < -255.0) right_pwm = -255.0;

  if (left_pwm > 0) {
    ledcWrite(M1_IN_1_CHANNEL, 0);
    ledcWrite(M1_IN_2_CHANNEL, (uint32_t)(left_pwm));
  } else {
    ledcWrite(M1_IN_1_CHANNEL, (uint32_t)-left_pwm);
    ledcWrite(M1_IN_2_CHANNEL, 0);
  }

  if (right_pwm > 0) {
    ledcWrite(M2_IN_1_CHANNEL, 0);
    ledcWrite(M2_IN_2_CHANNEL, (uint32_t)(right_pwm));
  } else {
    ledcWrite(M2_IN_1_CHANNEL, (uint32_t)-right_pwm);
    ledcWrite(M2_IN_2_CHANNEL, 0);
  }
}

float update_pid(float dt, float kp, float ki, float kd,
                 float x_d, float x,
                 float& int_e, float abs_int_e_max, // last_x and int_e are updated by this function
                 float& last_x) {
  // Calculate or update intermediates
  float e = x_d - x; // Error

  // Integrate error with anti-windup
  int_e = int_e + e * dt;
  if (int_e >  abs_int_e_max) int_e =  abs_int_e_max;
  if (int_e < -abs_int_e_max) int_e = -abs_int_e_max;

  // Take the "Derivative of the process variable" to avoid derivative spikes if setpoint makes step change
  // with abuse of notation, call this de
  float de = -(x - last_x) / dt;
  last_x = x;

  float u = kp * e + ki * int_e + kd * de;
  return u;
}

// a smooth and interesting trajectory
// https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli
void leminscate_of_bernoulli(float t, float a, float& x, float& y) {
  float sin_t = sin(t);
  float den = 1 + sin_t * sin_t;
  x = a * cos(t) / den;
  y = a * sin(t) * cos(t) / den;
}

// Signed angle from (x0, y0) to (x1, y1)
// assumes norms of these quantities are precomputed
float signed_angle(float x0, float y0, float n0, float x1, float y1, float n1) {
  float normed_dot = (x1 * x0 + y1 * y0) / (n1 * n0);
  if (normed_dot > 1.0) normed_dot = 1.0; // Possible because of numerical error
  float angle = acosf(normed_dot);

  // use cross product to find direction of rotation
  // https://en.wikipedia.org/wiki/Cross_product#Coordinate_notation
  float s3 = x0 * y1 - x1 * y0;
  if (s3 < 0) angle = -angle;

  return angle;
}



void setup() {
  Serial.begin(115200);

  // Disalbe the lightbar ADC chips so they don't hold the SPI bus used by the IMU
  pinMode(ADC_1_CS, OUTPUT);
  pinMode(ADC_2_CS, OUTPUT);
  digitalWrite(ADC_1_CS, HIGH);
  digitalWrite(ADC_2_CS, HIGH);
  delay(100);
  //Connect to the WiFi network
  connectToWiFi(networkName, networkPswd);

  ledcAttachPin(BUZZ, BUZZ_CHANNEL);

  pinMode(VCC_SENSE, INPUT);

  configure_motor_pins();
  configure_imu();

  adc1.begin(ADC_1_CS);
  adc2.begin(ADC_2_CS);
 

  Serial.println("Starting!");
  while (connected == false) {
    delay(100);
  }
}

void loop() {


  // Create the encoder objects after the motor has
  // stopped, else some sort exception is triggered
  Encoder enc1(M1_ENC_A, M1_ENC_B);
  Encoder enc2(M2_ENC_A, M2_ENC_B);
  
  // read in information from jetson
  
  // Loop period
  int target_period_ms = 2; // Loop takes about 3 ms so a delay of 2 gives 200 Hz or 5ms

  // States used to calculate target velocity and heading
  float leminscate_a = 0.5; // Radius
  float leminscate_t_scale = 2.0; // speedup factor
  float x0, y0;
  leminscate_of_bernoulli(0.0, leminscate_a, x0, y0);
  float last_x, last_y;
  leminscate_of_bernoulli(-leminscate_t_scale * target_period_ms / 1000.0, leminscate_a, last_x, last_y);
  float last_dx = (x0 - last_x) / ((float)target_period_ms / 1000.0);
  float last_dy = (y0 - last_y) / ((float)target_period_ms / 1000.0);
  float last_target_v = sqrtf(last_dx * last_dx + last_dy * last_dy);
  float target_theta = 0.0; // This is an integrated quantity

  // Motors are controlled by a position PID
  // with inputs interpreted in meters and outputs interpreted in volts
  // integral term has "anti-windup"
  // derivative term uses to derivative of process variable (wheel position)
  // instead of derivative of error in order to avoid "derivative kick"
  float kp_left = 200.0;
  float ki_left = 20.0;
  float kd_left = 20.0;
  float kf_left = 10.0;
  float target_pos_left  = 0.0;
  float last_pos_left = 0.0;
  float integral_error_pos_left = 0.0;
  float max_integral_error_pos_left = 1.0 * 8.0 / ki_left; // Max effect is the nominal battery voltage

  float kp_right = 200.0;
  float ki_right = 20.0;
  float kd_right = 20.0;
  float kf_right = 10.0;
  float last_pos_right = 0.0;
  float target_pos_right = 0.0;
  float integral_error_pos_right = 0.0;
  float max_integral_error_pos_right = 1.0 * 8.0 / ki_right; // Max effect is the nominal battery voltage

  // IMU Orientation variables
  float theta = 0.0;
  float bias_omega;
  // Gain applied to heading error when offseting target motor velocities
  // currently set to 360 deg/s compensation for 90 degrees of error
  float ktheta = (2 * 3.14159) / (90.0 * 3.14159 / 180.0);
  est_imu_bias(bias_omega, 500);// Could be expanded for more quantities

  // Global variables used for rotation, sound, and movement speed, etc
  // These are changed from wifi
  float phi = 0.00; 
  float sound = 0.00;
  float robot_phi = 0.00;

  // Control the logic and timing
  float moveStart = 0.00; 
  float backStart = 0.00;
  boolean moving = false;
  boolean movingBack = false;
  
  // arrays for adc sensors 
  int adc1_buf[8];
  int adc2_buf[8];

  // movement variables
  float theta_time = 0; // 1.95 for full 180
  float target_v = 0;
  float target_omega = 0;

  // determines the movement we will make
  int checkpoint_case = 0;  
  int theta_case = 0;   
  int count = 0;
  // The real "loop()"
  // time starts from 0
  float start_t = (float)micros() / 1000000.0;
  float last_t = -target_period_ms / 1000.0; // Offset by expected looptime to avoid divide by zero
  while (true) {
    // Get the time elapsed
    float t = ((float)micros()) / 1000000.0 - start_t;
    float dt = ((float)(t - last_t)); // Calculate time since last update
    
    // Wifi Loop
    
    if(connected) {
      //Send a packet
      udp.beginPacket(udpAddress,udpPort);
      udp.printf("Seconds since boot: %lu", millis()/1000);
      udp.endPacket();

      int packetSize = udp.parsePacket();

      if(packetSize >= 2*sizeof(float)) {
        udp.read((char*)&phi, sizeof(phi));
        udp.read((char*)&sound, sizeof(sound));
        udp.read((char*)&checkpoint_case, sizeof(checkpoint_case));
       // udp.read((char*)&checkpoint_case, sizeof(checkpoint_case)); 
        udp.flush();
        //Serial.printf("phi: %f\n", phi);
        //Serial.printf("sound: %f\n", sound);
      // Serial.printf("checkpoint: %f\n", checkpoint_case);
      } else { 
        Serial.println("Nothing to print");
      }
    }
      
      // can start the movement once we have valid phi and not moving
      if (phi != 0.00 && moving == false) {
        robot_phi += phi;
        count++;
        if(count == 20) { // averaged the first 20 phi values to eliminate an error with the first phi read
          moveStart = t;
          moving = true; // once we have 20 phi values, we can start moving
          robot_phi = robot_phi / 20;
        }
      }

    // determining the starting point of the robot 
    if(robot_phi >= 2.5 && robot_phi <= 3.5) { // starting point 1
      theta_case = 1;
    } else if (robot_phi >= 1.5 && robot_phi <= 2.2) { // starting point 2
      theta_case = 2; 
    } else if (robot_phi > 0.4 && robot_phi <= 1) { // starting point 3
      theta_case = 3;
    } else {
      theta_case = 0;
    }

    // 9 possible combinations of starting and end points
    switch (checkpoint_case) {
    case 1:
        switch (theta_case) {
            case 1:
                theta_time = 1.96;
                break;
            case 2:
                theta_time = 2.27;
                break;
            case 3:
                theta_time = 2.5;
                break;
            case 0: // if we don't yet have a phi reading
                theta_time = 0;
                break; 
        }
        break;
    case 2:
        switch (theta_case) {
            case 1:
                theta_time = 1.73;
                break;
            case 2:
                theta_time = 1.96;
                break;
            case 3:
                theta_time = 2.27;
                break;
            case 0:
                theta_time = 0;
                break;     
        }
        break;
    case 3:
        switch (theta_case) {
            case 1:
                theta_time = 1.5;
                break;
            case 2:
                theta_time = 1.69;
                break;
            case 3:
                theta_time = 1.96;
                break;
            case 0:
                theta_time = 0;
                break; 
        }
        break;
    }

    last_t = t;
    // Get the distances the wheels have traveled in meters
    // positive is forward
    float pos_left  =  (float)enc1.read() * METERS_PER_TICK;
    float pos_right = -(float)enc2.read() * METERS_PER_TICK; // Take negative because right counts upwards when rotating backwards
  
    // TODO Battery voltage compensation, the voltage sense on my mouse is broken for some reason
    // int counts = analogRead(VCC_SENSE);
    // float battery_voltage = counts * ADC_COUNTS_TO_VOLTS;
    // if (battery_voltage <= 0) Serial.println("BATTERY INVALID");
  
    // Read IMU and update estimate of heading
    // positive is counter clockwise
    float omega;
    read_imu(omega); // Could be expanded to read more things
    omega -= bias_omega; // Remove the constant bias measured in the beginning
    theta = theta + omega * dt;

    // Calculate target forward velocity and target heading to track the leminscate trajectory
    // of 0.5 meter radius
    float x, y;
    leminscate_of_bernoulli(leminscate_t_scale * t, leminscate_a, x, y);

    float dx = (x - last_x) / dt;
    float dy = (y - last_y) / dt;
    //float target_v = sqrtf(dx * dx + dy * dy); // forward velocity

    // Compute the change in heading using the normalized dot product between the current and last velocity vector
    // using this method instead of atan2 allows easy smooth handling of angles outsides of -pi / pi at the cost of
    // a slow drift defined by numerical precision
    // float target_omega = signed_angle(last_dx, last_dy, last_target_v, dx, dy, target_v) / dt;

    // Checkpoint and Robot Coordinates
    
    // assigning adc values to an array
    for (int i = 0; i < 8; i++) {
      adc1_buf[i] = adc1.readADC(i);
      adc2_buf[i] = adc2.readADC(i);
    }

    // setup for total adc_buf
    int adcT_buf[13];
    for(int i = 0; i < 7; i++){
      adcT_buf[i*2] = adc1_buf[i];
    }
    for(int i = 0; i < 6; i++){
      adcT_buf[i*2+1] = adc2_buf[i];
    }

    // movement starts if the boolean for the correct phi reading is triggered, and the initial movement lasts for
    // the duration of the calculated time for the angle needed
    if (((t-moveStart) < theta_time) && (moving == true)) {
      target_omega = 90.0 * (3.1459 / 180.0);
    } else if (moving == false) { // if moving bool is false, we don't move ever
      target_omega = 0;
      target_v = 0;
    } else { // if the other conditions are not met, we just move straight (turns have been completed)
      target_omega = 0;
      if (sound > 450) { // if at any time during our movement straight we hear a loud sound, we back up until its gone
        target_v = -0.1;
      } else {
        target_v = 0.3;
      } 
      if((adcT_buf[6] >= 680) && (adcT_buf[7] >= 680) && (adcT_buf[8] >= 680) && movingBack == false) { // if the middle 3 adc sensors sense black (start to turn)
        backStart = t; // set the time for the beginning of a turn
        movingBack = true; // into the last phase of the movement 
      }
    }
    // this is the movement back part
    if (((t-backStart) < 1.97) && (movingBack == true)) { // turn for 2 seconds (180 degrees back)
        target_v = 0;
        target_omega = 90.0 * (3.1459 / 180.0);
    }

    target_theta = target_theta + target_omega * dt; // uses the omega to make the heading

    // rest of the code below unchanged

    last_x = x;
    last_y = y;
    last_dx = dx;
    last_dy = dy;
    last_target_v = target_v;
  
    // Calculate target motor speeds from target forward speed and target heading
    // Could also include target path length traveled and target angular velocity
    float error_theta_z = target_theta - theta;
    float requested_v = target_v;
    float requested_w = ktheta * error_theta_z;

    float target_v_left  = requested_v - TURNING_RADIUS_METERS * requested_w;
    float target_v_right = requested_v + TURNING_RADIUS_METERS * requested_w;
    target_pos_left  = target_pos_left  + dt * target_v_left;
    target_pos_right = target_pos_right + dt * target_v_right;

    // Left motor position PID
    float left_voltage = update_pid(dt, kp_left, ki_left, kd_left,
                                    target_pos_left, pos_left,
                                    integral_error_pos_left, max_integral_error_pos_left,
                                    last_pos_left);
    left_voltage = left_voltage + kf_left * target_v_left;
    float left_pwm = (float)MAX_PWM_VALUE * (left_voltage / 8.0); // TODO use actual battery voltage

    // Right motor position PID
    float right_voltage = update_pid(dt, kp_right, ki_right, kd_right,
                                     target_pos_right, pos_right,
                                     integral_error_pos_right, max_integral_error_pos_right,
                                     last_pos_right);
    left_voltage = right_voltage + kf_right * target_v_right;
    float right_pwm = (float)MAX_PWM_VALUE * (right_voltage / 8.0); // TODO use actual battery voltage

    set_motors_pwm(left_pwm, right_pwm);

    delay(target_period_ms);
  }
}