/*
 *  ESP32  en mode station passive (scan)
 *  115200 output：SSID , BSSID , RSSI , Channel
 *  chaque scan ≈ 1.2 s
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
  // scan passive
  int n = WiFi.scanNetworks(false, true);   // 两个 true → 隐藏SSID也扫
  if (n == 0) {
    Serial.println("❌ Aucun AP detecte");
  } else {
    Serial.printf("✅ %d  AP detecte:\n", n);
    for (int i = 0; i < n; ++i) {
      String ssid     = WiFi.SSID(i);
      String bssid    = WiFi.BSSIDstr(i);   // forme 84:16:F9:AA:3C:21
      int32_t rssi    = WiFi.RSSI(i);
      uint8_t channel = WiFi.channel(i);

      // forme de output(json)
      Serial.printf("{\"ssid\":\"%s\", \"mac\":\"%s\", \"rssi\":%d, \"ch\":%d}\n",
                    ssid.c_str(),
                    bssid.c_str(),
                    rssi,
                    channel);

    }
  }

  WiFi.scanDelete();            // 释放内存
  delay(5000);                  // 每 5 s 重新扫一次
}