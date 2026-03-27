#include <Servo.h>

// --- Configuration & Pin Definitions ---
#define SERVO_ACTIVE 20  // 110
#define SERVO_INACTIVE (SERVO_ACTIVE + 90)
#define SERVO_1 3
#define SERVO_2 4

// Ultrasonic Sensor Pins
#define TRIG_PIN 22
#define ECHO_PIN 23

Servo s1, s2;

// Tank Calibration
float MIN_DISTANCE = 70.0;   // Distance (cm) when tank is empty
float MAX_DISTANCE = 5.0;    // Distance (cm) when tank is full
int MIN_LITERS = 0;           // Tank empty
int MAX_LITERS = 300;         // Tank full
float distance=0;
// State Variables
int tank_val = 0;

void setup() {
  Serial.begin(9600);

  s1.attach(SERVO_1);
  s2.attach(SERVO_2);
  s1.write(SERVO_ACTIVE);
  s2.write(SERVO_ACTIVE);

  delay(1000);
  
  s1.write(SERVO_INACTIVE);
  s2.write(SERVO_INACTIVE);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  updateTankLevel();  // Measure water level
  handleSerialInput(); // Read serial commands for servo control
  sendTelemetry();     // Send tank level over serial
  delay(100);          // Short delay for stability
}

// Function to measure tank water level
void updateTankLevel() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH);
  distance = duration * 0.034 / 2; // cm

  float liters = MAX_LITERS - mapFloat(distance, MAX_DISTANCE, MIN_DISTANCE, MIN_LITERS, MAX_LITERS);
  tank_val = (int)constrain(liters, MIN_LITERS, MAX_LITERS);
}

// Function to handle serial input to control servos
void handleSerialInput() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim(); // Remove whitespace

    if (line.startsWith("a:")) {
      int val = line.substring(2).toInt();
      s1.write(val == 0 ? SERVO_ACTIVE : SERVO_INACTIVE);
    }
    else if (line.startsWith("b:")) {
      int val = line.substring(2).toInt();
      s2.write(val == 0 ? SERVO_ACTIVE : SERVO_INACTIVE);
    }
  }
}

// Send tank level over Serial
void sendTelemetry() {
  Serial.print("tank_val:");
  Serial.println(tank_val);
  Serial.print("tank_min:");
  Serial.println(MIN_LITERS);
  Serial.print("tank_max:");
  Serial.println(MAX_LITERS);
  Serial.print("distance:");
  Serial.println(distance);
  Serial.print("percentage:");
  Serial.println(100*tank_val/MAX_LITERS);
  
}

// Helper function to map float values
float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}