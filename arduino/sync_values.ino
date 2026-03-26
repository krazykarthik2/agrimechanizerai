#include <Servo.h>

#define ULTRASONIC_ ECHO_PIN 2
#define ULTRASONIC_TRIG_PIN 3

#define SERVO_A 9
#define SERVO_B 10

Servo servoA,servoB;

void setup(){
    Serial.begin(9600); 
    pinMode(ULTRASONIC_ECHO_PIN, INPUT);
    pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
    pinMode(SERVO_A, OUTPUT);
    pinMode(SERVO_B, OUTPUT);

    servoA.attach(SERVO_A);
    servoB.attach(SERVO_B);

}

void loop(){


    if(Serial.available()){
        String input = Serial.readStringUntil(';');
        // parse by format a:a,b:b;
        int indexA = input.indexOf('a:');
        int indexB = input.indexOf('b:');
        int angleA = input.substring(indexA + 2, input.indexOf(',', indexA)).toInt();
        int angleB = input.substring(indexB + 2).toInt();

        // Write the angles to the servos
        servoA.write(angleA);
        servoB.write(angleB);
    }

    // Read the distance from the ultrasonic sensor
    long duration, distance;
    digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(ULTRASONIC_TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
    
    duration = pulseIn(ULTRASONIC_ECHO_PIN, HIGH);
    distance = (duration / 2) / 29.1; // Convert to cm


    Serial.print("tank:");
    Serial.println(distance);
    Serial.print("a:");
    Serial.println(angleA);
    Serial.print("b:");
    Serial.println(angleB);

    delay(50); // Delay for stability
}