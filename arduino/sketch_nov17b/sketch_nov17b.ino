/*
 *  ESP32 + LoRa-E5
 *  Étape 2 : WiFi sniffing + transmission LoRaWAN (TTN)
 *  - Scan WiFi
 *  - Build compact JSON
 *  - Send via LoRa-E5 (AT commands)
 *  - JOIN retry + protection
 */

#include <WiFi.h>
#include <HardwareSerial.h>

/* ---------- LoRa UART ---------- */
HardwareSerial LoraSerial(2);   // RX=16, TX=17

/* ---------- TTN parameters ---------- */
const char *app_eui = "3345979272562857";
const char *dev_eui = "70B3D57ED007398E";
const char *app_key = "C4663E9406A879F44EE74D93D977A14D";

/* ---------- Config ---------- */
const int MAX_AP = 20;                     // max APs per scan (more may require fragmentation)
const int APS_PER_PACKET = 4;              // APs per LoRa message (controls payload size)
const unsigned long TX_INTERVAL = 30000;   // 30 s
const unsigned long FRAG_DELAY_MS = 5000;  // delay between fragments (EU868 duty cycle friendly)
unsigned long last_tx = 0;


/* ---------- JOIN retry config ---------- */
const int JOIN_RETRY_MAX = 5;
const unsigned long JOIN_RETRY_DELAY = 5000;

/* ---------- Global state ---------- */
bool lora_joined = false;

/* ---------- Prototypes ---------- */
void send_scan_fragmented();
void init_lora_e5();
bool join_lora();
void send_lora_packet(const String &json);
bool send_at(String cmd, int wait = 1000);

/* ---------- SETUP ---------- */
void setup() {
  Serial.begin(115200);
  delay(1000);

  /* WiFi init */
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(100);

  /* LoRa UART */
  LoraSerial.begin(9600, SERIAL_8N1, 16, 17);

  Serial.println("\n----- ESP32 WiFi Sniffer + LoRaWAN -----");

  init_lora_e5();

  /* JOIN with retry */
  bool joined = false;
  int retry = 0;

  while (!joined && retry < JOIN_RETRY_MAX) {
    Serial.printf("🔄 JOIN attempt %d / %d\n", retry + 1, JOIN_RETRY_MAX);

    joined = join_lora();

    if (!joined) {
      retry++;
      Serial.println("❌ JOIN failed, retrying...");
      delay(JOIN_RETRY_DELAY);
    }
  }

  if (!joined) {
    Serial.println("🚨 Unable to join LoRaWAN network");
  } else {
    Serial.println("✅ LoRaWAN JOIN success");
    lora_joined = true;
  }
}

/* ---------- LOOP ---------- */
void loop() {
  if (!lora_joined) {
    delay(1000);
    return;
  }

  if (millis() - last_tx > TX_INTERVAL) {
    send_scan_fragmented();  // one scan, send as multiple packets
    last_tx = millis();
  }
}



/* ---------- WiFi scan + JSON ---------- */

String json_escape(const String &in) {
  String out = "";
  for (size_t i = 0; i < in.length(); i++) {
    char c = in[i];
    if (c == '\"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else out += c;
  }
  return out;
}


void send_scan_fragmented() {
  int n = WiFi.scanNetworks(false, true);
  if (n <= 0) {
    Serial.println("No AP found");
    return;
  }

  n = min(n, MAX_AP);

  unsigned long ts = millis() / 1000;
  unsigned long sid = millis();

  int total = (n + APS_PER_PACKET - 1) / APS_PER_PACKET;

  for (int p = 0; p < total; p++) {
    int start = p * APS_PER_PACKET;
    int end = min(start + APS_PER_PACKET, n);

    String json = "{\"sid\":" + String(sid) +
                  ",\"i\":" + String(p) +
                  ",\"n\":" + String(total) +
                  ",\"t\":" + String(ts) +
                  ",\"a\":[";

    for (int k = start; k < end; k++) {
      if (k > start) json += ",";

      String ssid = json_escape(WiFi.SSID(k)); // may be empty string

      json += "[\"";
      json += WiFi.BSSIDstr(k);
      json += "\",";
      json += String(WiFi.RSSI(k));
      json += ",";
      json += String(WiFi.channel(k));
      json += ",\"";
      json += ssid;
      json += "\"]";
    }

    json += "]}";

    Serial.println("JSON: " + json);
    send_lora_packet(json);
    delay(FRAG_DELAY_MS);
  }

  WiFi.scanDelete();
}





/* ---------- LoRa init ---------- */
void init_lora_e5() {
  send_at("AT");
  send_at("AT+MODE=LWOTAA");

  send_at(String("AT+ID=AppEui,\"") + app_eui + "\"");
  send_at(String("AT+ID=DevEui,\"") + dev_eui + "\"");
  send_at(String("AT+KEY=APPKEY,\"") + app_key + "\"");

  send_at("AT+DR=EU868");   // region
  send_at("AT+DR=5");       // DR5 = SF7 (max payload)
  send_at("AT+CLASS=A");
}

/* ---------- JOIN ---------- */
bool join_lora() {
  LoraSerial.println("AT+JOIN");
  Serial.println("> AT+JOIN");

  unsigned long start = millis();
  while (millis() - start < 10000) {
    if (LoraSerial.available()) {
      String r = LoraSerial.readString();
      Serial.print(r);

      if (r.indexOf("Network joined") >= 0) return true;
      if (r.indexOf("JOIN failed") >= 0) return false;
    }
  }
  return false;
}

/* ---------- Send packet ---------- */
void send_lora_packet(const String &json) {
  String hex = "";
  for (int i = 0; i < json.length(); i++) {
    byte c = json[i];
    if (c < 16) hex += "0";
    hex += String(c, HEX);
  }
  hex.toUpperCase();

  String cmd = "AT+MSGHEX=\"" + hex + "\"";
  send_at(cmd, 3000);
}

/* ---------- AT helper ---------- */
bool send_at(String cmd, int wait) {
  LoraSerial.println(cmd);
  Serial.println("> " + cmd);
  delay(wait);

  while (LoraSerial.available()) {
    Serial.write(LoraSerial.read());
  }
  Serial.println();
  return true;
}
