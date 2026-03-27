#include <SoftwareSerial.h>

#define LEFT_MAX_THETA 180
#define LEFT_MIN_THETA 0
#define RIGHT_MAX_THETA 180
#define RIGHT_MIN_THETA 0

SoftwareSerial mySerial(10, 11); // RX, TX

void setup() {
  Serial.begin(9600);      // USB Serial (PC)
  mySerial.begin(9600);    // Other device
}

void loop() {
  int x = analogRead(A0);
  int y = analogRead(A1);

  int thetaLeft = map(x, 0, 1023, LEFT_MIN_THETA, LEFT_MAX_THETA);
  int thetaRight = map(y, 0, 1023, RIGHT_MIN_THETA, RIGHT_MAX_THETA);

  Serial.print("left:");
  Serial.println(thetaLeft);
  Serial.print("right:");
  Serial.println(thetaRight);

  // Serial.print(" ");
  // Serial.print("A0:");
  // Serial.print(x);
  // Serial.print(" ");
  // Serial.print("A1:");
  // Serial.println(y);


  // Two-way communication
  if (mySerial.available()) {
    char c = mySerial.read();
    Serial.write(c);   // forward to PC
  }

  if (Serial.available()) {
    char c = Serial.read();
    mySerial.write(c); // forward to other device
  }

  delay(20);
}