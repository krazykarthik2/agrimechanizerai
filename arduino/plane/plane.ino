#include <Servo.h>

// --- Configuration & Pin Definitions ---
#define SERVO_ACTIVE 20     // Position to STOP spraying
#define SERVO_INACTIVE 110  // Position to START spraying
#define SERVO_1 3           // Left Side
#define SERVO_2 4           // Right Side

// Ultrasonic Sensor Pins
#define TRIG_PIN 22
#define ECHO_PIN 23

Servo s1, s2;

// Tank Calibration
float MIN_DISTANCE = 70.0;    // Distance (cm) when tank is empty
float MAX_DISTANCE = 5.0;     // Distance (cm) when tank is full
int MIN_LITERS = 0;           
int MAX_LITERS = 300;         
float distance = 0;
int tank_val = 0;

void setup() {
  Serial.begin(9600);

  s1.attach(SERVO_1);
  s2.attach(SERVO_2);
  
  // Initialization sequence
  s1.write(SERVO_ACTIVE);
  s2.write(SERVO_ACTIVE);
  delay(1000);
  
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  updateTankLevel();   
  handleSerialInput(); 
  sendTelemetry();     
  delay(50); // Increased frequency for better response
}

void updateTankLevel() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  distance = duration * 0.034 / 2; 

  if (distance > 0) {
    float liters = mapFloat(distance, MAX_DISTANCE, MIN_DISTANCE, MAX_LITERS, MIN_LITERS);
    tank_val = (int)constrain(liters, MIN_LITERS, MAX_LITERS);
  }
}

/**
 * UPDATED: Parses "write:L;R"
 * Example: "write:1;0" sets Left ON and Right OFF
 */
void handleSerialInput() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("write:")) {
      // Find the positions of ':' and ';'
      int colonIdx = line.indexOf(':');
      int semiIdx = line.indexOf(';');

      if (colonIdx != -1 && semiIdx != -1) {
        // Extract Left bit (between : and ;)
        int leftBit = line.substring(colonIdx + 1, semiIdx).toInt();
        // Extract Right bit (after ;)
        int rightBit = line.substring(semiIdx + 1).toInt();

        // Control Servos: 1 = Inactive (Spray), 0 = Active (Stop)
        // Adjust these if your mechanics are reversed
        s1.write(leftBit == 1 ? SERVO_INACTIVE : SERVO_ACTIVE);
        s2.write(rightBit == 1 ? SERVO_INACTIVE : SERVO_ACTIVE);
      }
    }
  }
}

/**
 * UPDATED: Sends "read:val,cap;"
 * This matches the Python line: parts = line[5:-1].split(',')
 */
void sendTelemetry() {
  Serial.print("read:");
  Serial.print(tank_val);
  Serial.print(",");
  Serial.print(MAX_LITERS);
  Serial.println(";");
}

float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}