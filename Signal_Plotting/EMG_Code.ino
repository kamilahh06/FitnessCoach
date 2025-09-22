#include <SPI.h>

int CS_PIN = 5;

int readADC(int channel) {
  byte command = 0b11000000 | (channel << 3);
  digitalWrite(CS_PIN, LOW);
  byte high = SPI.transfer(command);
  byte low = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  int value = ((high & 0x0F) << 8) | low;
  return value;
}

void setup() {
  Serial.begin(115200);
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
}

void loop() {
  int val = readADC(0);
  Serial.print(millis());
  Serial.print(",");
  Serial.println(val);
  delay(100);
}
