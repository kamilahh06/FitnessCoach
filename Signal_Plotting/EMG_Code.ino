#include <SPI.h>

int CS_PIN = 5;
unsigned long lastMicros = 0;
const unsigned long sampleInterval = 5000;

int readADC(int channel) {
  byte command1 = 0b00000001; 
  byte command2 = 0b1000 << 4 | (channel << 4); 
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(command1);
  int high = SPI.transfer(command2) & 0x03; 
  int low = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  return (high << 8) | low;
}

void setup() {
  Serial.begin(115200);
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
}

void loop() {
  unsigned long now = micros();
  if (now - lastMicros >= sampleInterval) {
    lastMicros = now;

    int val = readADC(0);          
    unsigned long t = millis();    
    Serial.print(t);
    Serial.print(",");
    Serial.println(val);
  }
}
