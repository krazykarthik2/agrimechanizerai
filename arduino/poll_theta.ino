
#define LEFT_MAX_THETA 180
#define LEFT_MIN_THETA 0
#define RIGHT_MAX_THETA 180
#define RIGHT_MIN_THETA 0

void setup(){
    Serial.begin(9600);
    // A0, A1 as input
    pinMode(A0, INPUT);
    pinMode(A1, INPUT);
}
void loop(){
    int x = analogRead(A0);
    int y = analogRead(A1);

    int thetaleft = map(x, 0, 1023, LEFT_MIN_THETA, LEFT_MAX_THETA);
    int thetaright = map(y, 0, 1023, RIGHT_MIN_THETA, RIGHT_MAX_THETA);
    Serial.print("left:");
    Serial.println(thetaleft);
    Serial.print("right:");
    Serial.println(thetaright);
    delay(50);
}