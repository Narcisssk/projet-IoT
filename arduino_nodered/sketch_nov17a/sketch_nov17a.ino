/*
 *  ESP32 in station mode, passive scan
 *  115200 baud output: SSID, BSSID, RSSI, Channel
 *  each scan ~1.2 s
 */

#include <WiFi.h>

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);          // mode station passive
  WiFi.disconnect();            // wifi connect off
  delay(100);

  Serial.println("\n-----  ESP32 WiFi Sniffer  -----");
}

void loop() {
  // passive scan
  int n = WiFi.scanNetworks(false, true);   // (false, true) = also scan hidden SSIDs
  if (n == 0) {
    Serial.println("❌ Aucun AP detecte");
  } else {
    Serial.printf("✅ %d  AP detecte:\n", n);
    for (int i = 0; i < n; ++i) {
      String ssid     = WiFi.SSID(i);
      String bssid    = WiFi.BSSIDstr(i);   // format e.g. 84:16:F9:AA:3C:21
      int32_t rssi    = WiFi.RSSI(i);
      uint8_t channel = WiFi.channel(i);

      // output format (JSON)
      Serial.printf("{\"ssid\":\"%s\", \"mac\":\"%s\", \"rssi\":%d, \"ch\":%d}\n",
                    ssid.c_str(),
                    bssid.c_str(),
                    rssi,
                    channel);

    }
  }

  WiFi.scanDelete();            // free scan buffer
  delay(5000);                  // rescan every 5 s
}