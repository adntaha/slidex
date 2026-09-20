#include <Arduino.h>
#include "ESP_I2S.h"
#include "BluetoothSerial.h"

I2SClass I2S;
BluetoothSerial SerialBT;

// Your verified microphone pins
#define I2S_BCLK 26
#define I2S_LRCL 25
#define I2S_DOUT 39
#define SAMPLE_RATE 16000

// Button Input Pins
#define PAUSE 13
#define WIPE  2   
#define NEXT  15

// Audio Packet Configuration
#define AUDIO_CHUNK_SIZE 256  // Send 256 samples at a time for efficiency
int16_t audioPayload[AUDIO_CHUNK_SIZE]; 
int16_t sampleIndex = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(WIPE, INPUT_PULLUP);
  pinMode(NEXT, INPUT_PULLUP);
  pinMode(PAUSE, INPUT_PULLUP);

  SerialBT.begin("ESP32_Speech_Link");
  Serial.println("Starting ICS-43434...");

  I2S.setPins(I2S_BCLK, I2S_LRCL, -1, I2S_DOUT, -1);

  // We change the width to 16BIT inside the container to maximize Bluetooth bandwidth
  // 16-bit Mono at 16kHz is standard studio input for Whisper/Vosk speech models
  if (!I2S.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("I2S initialization failed!");
    while (true) { delay(1000); }
  }

  Serial.println("ICS-43434 and Bluetooth Data Link Ready!");
}

void loop() {
  int16_t rawSample = 0;

  // Read a single 16-bit microphone sample
  size_t bytesRead = I2S.readBytes((char *)&rawSample, sizeof(rawSample));

  if (bytesRead == sizeof(rawSample)) {
    // Add sample to our growing batch array
    audioPayload[sampleIndex] = rawSample;
    sampleIndex++;

    // When the batch array fills up, send it out as a structured frame
    if (sampleIndex >= AUDIO_CHUNK_SIZE) {
      
      if (SerialBT.hasClient()) {
        // 1. Send unique 4-byte frame header so the PC can sync up
        uint8_t header[4] = {0xAA, 0xBB, 0xCC, 0xDD};
        SerialBT.write(header, 4);

        // 2. Read the hardware button states right now
        uint8_t buttonState = 0x00;
        if (digitalRead(WIPE) == LOW)  buttonState |= 0x01; // Bit 0
        if (digitalRead(PAUSE) == LOW) buttonState |= 0x02; // Bit 1
        if (digitalRead(NEXT) == LOW)  buttonState |= 0x04; // Bit 2
        
        // 3. Send the single-byte button snapshot flags
        SerialBT.write(buttonState);

        // 4. Send the 512 bytes of raw, un-noised PCM audio track payload
        SerialBT.write((uint8_t*)audioPayload, AUDIO_CHUNK_SIZE * sizeof(int16_t));
      }

      // Reset block index counter for next wave
      sampleIndex = 0;
    }
  }
}
