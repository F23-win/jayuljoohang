const long BAUD_RATE = 115200;
const unsigned long SAFETY_TIMEOUT_MS = 500;

// 실제 배선
// 오른쪽 뒷바퀴: IN1=D3, IN2=D4
// 왼쪽 뒷바퀴: IN1=D7, IN2=D8
// 핸들: IN1=D11, IN2=D12
const int RIGHT_DRIVE_IN1_PIN = 3;
const int RIGHT_DRIVE_IN2_PIN = 4;
const int LEFT_DRIVE_IN1_PIN = 7;
const int LEFT_DRIVE_IN2_PIN = 8;
const int STEER_IN1_PIN = 11;
const int STEER_IN2_PIN = 12;

// 바퀴가 서로 반대로 돌면 해당 값을 true로 바꾸세요.
const bool RIGHT_DRIVE_INVERT = true;
const bool LEFT_DRIVE_INVERT = true;
const bool STEERING_INVERT = false;

unsigned long lastCommandAt = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(RIGHT_DRIVE_IN1_PIN, OUTPUT);
  pinMode(RIGHT_DRIVE_IN2_PIN, OUTPUT);
  pinMode(LEFT_DRIVE_IN1_PIN, OUTPUT);
  pinMode(LEFT_DRIVE_IN2_PIN, OUTPUT);
  pinMode(STEER_IN1_PIN, OUTPUT);
  pinMode(STEER_IN2_PIN, OUTPUT);
  stopAll();
  lastCommandAt = millis();
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    handleCommand(line);
    lastCommandAt = millis();
  }

  if (millis() - lastCommandAt > SAFETY_TIMEOUT_MS) {
    stopAll();
  }
}

void handleCommand(String line) {
  if (line == "STOP") {
    stopAll();
    return;
  }

  if (line == "PING") {
    Serial.println("PONG");
    return;
  }

  if (line.startsWith("DRIVE ")) {
    int firstSpace = line.indexOf(' ');
    int secondSpace = line.indexOf(' ', firstSpace + 1);
    if (secondSpace < 0) {
      stopAll();
      return;
    }

    int speed = constrain(line.substring(firstSpace + 1, secondSpace).toInt(), -255, 255);
    int steering = constrain(line.substring(secondSpace + 1).toInt(), -255, 255);

    int rightSpeed = RIGHT_DRIVE_INVERT ? -speed : speed;
    int leftSpeed = LEFT_DRIVE_INVERT ? -speed : speed;
    int steeringValue = STEERING_INVERT ? -steering : steering;

    applyMotor(RIGHT_DRIVE_IN1_PIN, RIGHT_DRIVE_IN2_PIN, rightSpeed);
    applyMotor(LEFT_DRIVE_IN1_PIN, LEFT_DRIVE_IN2_PIN, leftSpeed);
    applyMotor(STEER_IN1_PIN, STEER_IN2_PIN, steeringValue);
  }
}

void applyMotor(int in1, int in2, int value) {
  int pwm = abs(value);
  if (value > 0) {
    analogWrite(in1, pwm);
    analogWrite(in2, 0);
  } else if (value < 0) {
    analogWrite(in1, 0);
    analogWrite(in2, pwm);
  } else {
    analogWrite(in1, 0);
    analogWrite(in2, 0);
  }
}

void stopAll() {
  applyMotor(RIGHT_DRIVE_IN1_PIN, RIGHT_DRIVE_IN2_PIN, 0);
  applyMotor(LEFT_DRIVE_IN1_PIN, LEFT_DRIVE_IN2_PIN, 0);
  applyMotor(STEER_IN1_PIN, STEER_IN2_PIN, 0);
}
