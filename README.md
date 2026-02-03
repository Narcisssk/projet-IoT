# TP3 – WiFi Sniffing & Geolocation (LoRaWAN)

## Project Layout

| Folder      | Contents |
|------------|----------|
| **data/**  | CSV and JSONL data files |
| **scripts/** | Python scripts for positioning and simulation |
| **notebooks/** | Jupyter notebook for data processing |
| **arduino/** | ESP32 firmware sketches |
| **docs/**  | Project documentation (PDF) |

## Code Files Overview

### Arduino Firmware (`arduino/`)

#### `sketch_nov17a/sketch_nov17a.ino`
- **Purpose**: WiFi scanning only (testing/debugging)
- **Function**: ESP32 scans nearby WiFi APs in passive mode, outputs SSID, BSSID, RSSI, and channel via Serial (115200 baud)
- **Output**: JSON format per AP: `{"ssid":"...", "mac":"...", "rssi":..., "ch":...}`
- **Use case**: Verify WiFi scanning works before adding LoRaWAN transmission

#### `sketch_nov17b/sketch_nov17b.ino`
- **Purpose**: WiFi scanning + LoRaWAN transmission to The Things Network (TTN)
- **Function**: 
  - Scans WiFi APs (up to `MAX_AP` per scan)
  - Builds compact JSON payload with session ID, fragment index, timestamp, and AP list
  - Fragments large scans into multiple LoRa packets (4 APs per packet)
  - Sends via LoRa-E5 module using AT commands
  - Handles JOIN retry logic and duty cycle compliance (EU868)
- **Output**: LoRaWAN uplinks to TTN → MQTT → FastAPI server (see notebook)
- **Use case**: Production firmware for field deployment

### Jupyter Notebook (`notebooks/tp3.ipynb`)

The notebook contains multiple cells for the data pipeline:

#### Cell 1: TTN MQTT Listener + FastAPI Server
- **Purpose**: Receive LoRaWAN uplinks from TTN via MQTT and store to CSV
- **Function**: 
  - Subscribes to TTN MQTT broker (`v3/+/devices/+/up`)
  - Parses uplink payloads (BSSID, RSSI, channel, SSID)
  - Appends to `data/wifi_sniffing_data.csv`
  - Provides FastAPI health endpoint
- **Output**: `data/wifi_sniffing_data.csv` (raw WiFi observations)

#### Cell 3: Wigle API Enrichment
- **Purpose**: Query Wigle database to get GPS coordinates for WiFi APs
- **Function**:
  - Extracts unique (BSSID, SSID) pairs from raw CSV
  - Queries Wigle API for each pair (limited to Paris region)
  - Filters results by SSID match and geographic bounds
  - Builds AP database with lat/lon coordinates
- **Output**: `data/ap_db_wigle.csv` (AP database with real-world coordinates)

#### Cell 4: Synthetic AP Database Generator
- **Purpose**: Generate fake AP coordinates for testing (when Wigle data unavailable)
- **Function**:
  - Reads BSSIDs from raw CSV
  - Assigns random (lat, lon) coordinates near a center point (within radius)
  - Uses fixed random seed for reproducibility
- **Output**: `data/ap_db_fake.csv` (synthetic AP database)

#### Cell 6: Simulated Uplink Generator
- **Purpose**: Generate synthetic TTN uplink messages for algorithm testing
- **Function**:
  - Loads AP database (fake or Wigle)
  - Generates random device positions within radius
  - Simulates RSSI using log-distance path loss model + noise
  - Builds TTN-compatible JSONL format
- **Output**: `data/sim_uplinks.jsonl` (simulated uplinks for testing)

### Python Scripts (`scripts/`)

#### `distance.py`
- **Purpose**: Compare three positioning algorithms using WiFi RSSI observations
- **Algorithms**:
  1. **Weighted Centroid**: RSSI-weighted average of AP positions
  2. **Multilateration**: RSSI → distance conversion, then least-squares trilateration (Gauss-Newton)
  3. **KNN Fingerprint**: Machine learning approach using synthetic training data
- **Function**:
  - Reads AP database and simulated uplinks
  - Estimates position using each algorithm
  - Computes RSSI RMSE for each estimate
  - Ranks methods and generates comparison report
- **Output**: `data/position_estimates_compare.csv` (results with rankings and statistics)
- **Usage**: `python scripts/distance.py`

#### `sim_geo_publisher.py`
- **Purpose**: Publish simulated uplinks to TTN MQTT broker (for testing end-to-end pipeline)
- **Function**:
  - Reads JSONL file with simulated uplinks
  - Loads AP database for multilateration fallback
  - Publishes each uplink to TTN MQTT broker
  - Can add ground truth GPS coordinates to payload
- **Usage**: `python scripts/sim_geo_publisher.py --jsonl data/sim_uplinks.jsonl --apdb data/ap_db_fake.csv`

## Data Files (`data/`)

- `wifi_sniffing_data.csv`: Raw WiFi observations from ESP32 (via TTN)
- `ap_db_wigle.csv`: AP database with real GPS coordinates (from Wigle API)
- `ap_db_fake.csv`: Synthetic AP database (for testing)
- `ap_db_template.csv`: Template/example AP database format
- `sim_uplinks.jsonl`: Simulated TTN uplink messages (for algorithm testing)
- `position_estimates_compare.csv`: Positioning algorithm comparison results

