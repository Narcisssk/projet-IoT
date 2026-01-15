import json
import time
import csv
import argparse
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

EARTH_R = 6378137.0  # WGS84 sphere approx


# -----------------------------
# Geo: lat/lon <-> local meters (small area)
# -----------------------------
def ll_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    phi = math.radians(lat)
    phi0 = math.radians(lat0)
    lam = math.radians(lon)
    lam0 = math.radians(lon0)
    x = (lam - lam0) * math.cos((phi + phi0) / 2.0) * EARTH_R
    y = (phi - phi0) * EARTH_R
    return x, y


def xy_to_ll(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    phi0 = math.radians(lat0)
    lam0 = math.radians(lon0)
    phi = y / EARTH_R + phi0
    lam = x / (EARTH_R * math.cos((phi + phi0) / 2.0)) + lam0
    return math.degrees(phi), math.degrees(lam)


# -----------------------------
# RSSI -> distance (log-distance path loss)
# d = d0 * 10^((RSSI0 - RSSI)/(10*n))
# -----------------------------
def distance_from_rssi(rssi_dbm: float, rssi0_dbm: float = -45.0, d0_m: float = 1.0, n: float = 2.3) -> float:
    # Clamp to avoid absurd ranges due to outliers
    rssi = float(rssi_dbm)
    d = d0_m * (10.0 ** ((rssi0_dbm - rssi) / (10.0 * n)))
    return max(0.3, min(d, 500.0))


# -----------------------------
# Load AP DB (columns: bssid,lat,lon,source)
# -----------------------------
def load_ap_db(csv_path: str) -> Dict[str, Dict[str, Any]]:
    apdb: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.lower() for c in (reader.fieldnames or [])]
        if "bssid" not in cols or "lat" not in cols or "lon" not in cols:
            raise ValueError("AP DB must have columns: bssid,lat,lon,source(optional)")

        def get_ci(row, name: str) -> Optional[str]:
            for k in row.keys():
                if k.lower() == name:
                    return row.get(k)
            return None

        for row in reader:
            bssid = (get_ci(row, "bssid") or "").strip().lower()
            if not bssid:
                continue
            try:
                lat = float(get_ci(row, "lat"))
                lon = float(get_ci(row, "lon"))
            except Exception:
                continue
            source = (get_ci(row, "source") or "").strip()
            apdb[bssid] = {"lat": lat, "lon": lon, "source": source}
    return apdb


# -----------------------------
# Fallback: RSSI-weighted centroid (only used if ML cannot run)
# -----------------------------
def rssi_weight_centroid(rssi: float) -> float:
    return max(1.0, 100.0 + float(rssi))


def estimate_weighted_centroid(used: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    sw = slat = slon = 0.0
    for ap in used:
        rssi = float(ap["rssi"])
        w = rssi_weight_centroid(rssi)
        sw += w
        slat += w * float(ap["lat"])
        slon += w * float(ap["lon"])
    if sw <= 0:
        return None, None
    return slat / sw, slon / sw


# -----------------------------
# ML: multilateration (Gauss-Newton in local XY)
# Returns (lat,lon,rms_m) or (None,None,None)
# -----------------------------
def estimate_multilateration_xy(
    pts_xy: List[Tuple[float, float]],
    dists_m: List[float],
    max_iter: int = 25,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(pts_xy) < 3:
        return None, None, None

    # init at centroid of AP positions
    x = sum(p[0] for p in pts_xy) / len(pts_xy)
    y = sum(p[1] for p in pts_xy) / len(pts_xy)

    for _ in range(max_iter):
        # build residuals & Jacobian
        J11 = J12 = J22 = 0.0
        b1 = b2 = 0.0

        for (xi, yi), di in zip(pts_xy, dists_m):
            dx = x - xi
            dy = y - yi
            ri = math.hypot(dx, dy)
            ri = max(ri, 1e-6)

            # residual: (ri - di)
            f = (ri - di)

            # jacobian: [dx/ri, dy/ri]
            jx = dx / ri
            jy = dy / ri

            # accumulate normal equations: (J^T J) delta = -J^T f
            J11 += jx * jx
            J12 += jx * jy
            J22 += jy * jy
            b1 += jx * f
            b2 += jy * f

        # damping for stability
        lam = 1e-3
        A11 = J11 + lam
        A12 = J12
        A21 = J12
        A22 = J22 + lam

        det = A11 * A22 - A12 * A21
        if abs(det) < 1e-12:
            break

        # delta = - inv(A) * b
        dx = -(A22 * b1 - A12 * b2) / det
        dy = -(-A21 * b1 + A11 * b2) / det

        x += dx
        y += dy

        if math.hypot(dx, dy) < 1e-3:
            break

    # compute RMS of range residuals
    errs = []
    for (xi, yi), di in zip(pts_xy, dists_m):
        ri = math.hypot(x - xi, y - yi)
        errs.append((ri - di) ** 2)
    rms = math.sqrt(sum(errs) / len(errs)) if errs else None

    return x, y, rms


# -----------------------------
# Main estimator wrapper (keeps your Node-RED interface unchanged)
# Input aps_rx items: [bssid, rssi, channel, ssid]
# Output: (est_lat, est_lon, used_aps, method_string)
# -----------------------------
def estimate_position_ml(
    apdb: Dict[str, Dict[str, Any]],
    aps_rx: List[List[Any]],
    lat0: float,
    lon0: float,
    rssi0_dbm: float,
    n_pathloss: float,
) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]], str]:

    used: List[Dict[str, Any]] = []
    pts_xy: List[Tuple[float, float]] = []
    dists: List[float] = []

    for item in aps_rx:
        if not isinstance(item, list) or len(item) < 2:
            continue

        bssid = str(item[0]).strip().lower()
        try:
            rssi = float(item[1])
        except Exception:
            continue

        ch = item[2] if len(item) > 2 else None
        ssid_obs = item[3] if len(item) > 3 else ""

        if bssid not in apdb:
            continue

        lat = apdb[bssid]["lat"]
        lon = apdb[bssid]["lon"]
        src = apdb[bssid].get("source", "")

        used.append({
            "bssid": bssid,
            "mac": bssid,  # for UI compatibility
            "rssi": rssi,
            "ch": ch,
            "ssid_obs": ssid_obs,
            "lat": lat,
            "lon": lon,
            "source": src
        })

        x, y = ll_to_xy_m(lat, lon, lat0, lon0)
        pts_xy.append((x, y))
        dists.append(distance_from_rssi(rssi, rssi0_dbm=rssi0_dbm, n=n_pathloss))

    # ML needs >= 3 APs
    if len(used) >= 3:
        x_hat, y_hat, _rms = estimate_multilateration_xy(pts_xy, dists)
        if x_hat is not None and y_hat is not None:
            est_lat, est_lon = xy_to_ll(x_hat, y_hat, lat0, lon0)
            return est_lat, est_lon, used, "multilateration"

    # Fallback if not enough APs or solver failed
    est_lat, est_lon = estimate_weighted_centroid(used) if used else (None, None)
    return est_lat, est_lon, used, "weighted_centroid_fallback"


class Reassembler:
    """
    Reassemble fragments by sid.
    Accept i either 0-based or 1-based.
    """
    def __init__(self):
        self.buf: Dict[str, Dict[str, Any]] = {}

    def push(self, frag: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sid = str(frag.get("sid"))
        i = int(frag.get("i", 0))
        n = int(frag.get("n", 1))
        t_unix = frag.get("t")
        a = frag.get("a") or []

        # normalize 1-based -> 0-based
        if 1 <= i <= n:
            i = i - 1

        e = self.buf.setdefault(sid, {"n": n, "t": t_unix, "parts": {}, "count": 0})
        if i not in e["parts"]:
            e["parts"][i] = a
            e["count"] += 1
        e["n"] = n
        e["t"] = t_unix

        if e["count"] >= e["n"]:
            aps: List[Any] = []
            for k in range(e["n"]):
                aps.extend(e["parts"].get(k, []))
            del self.buf[sid]
            return {"sid": sid, "t_unix": e["t"], "aps_rx": aps}
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="sim_uplinks.jsonl")
    ap.add_argument("--apdb", required=True, help="ap db csv (bssid,lat,lon,source)")
    ap.add_argument("--broker", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--username", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--topic_out", default="sim/geo_result")
    ap.add_argument("--min_sleep", type=float, default=2.0)
    ap.add_argument("--max_sleep", type=float, default=6.0)
    ap.add_argument("--loop", action="store_true")

    # ML model params (keep defaults consistent with your simulation)
    ap.add_argument("--rssi0", type=float, default=-45.0, help="RSSI at 1m (dBm)")
    ap.add_argument("--n", type=float, default=2.3, help="Path loss exponent")

    args = ap.parse_args()

    apdb = load_ap_db(args.apdb)

    # projection origin: mean of all APs (stable)
    lat0 = sum(v["lat"] for v in apdb.values()) / max(1, len(apdb))
    lon0 = sum(v["lon"] for v in apdb.values()) / max(1, len(apdb))

    c = mqtt.Client(client_id="sim-geo-publisher")
    if args.username is not None:
        c.username_pw_set(args.username, args.password or "")
    c.connect(args.broker, args.port, 60)
    c.loop_start()

    reasm = Reassembler()

    def publish_geo(envelope: Dict[str, Any], scan_complete: Dict[str, Any]):
        device_id = (envelope.get("end_device_ids") or {}).get("device_id", "unknown")
        received_at = envelope.get("received_at")

        sid = scan_complete["sid"]
        t_unix = scan_complete["t_unix"]
        aps_rx = scan_complete["aps_rx"]

        est_lat, est_lon, used, method = estimate_position_ml(
            apdb=apdb,
            aps_rx=aps_rx,
            lat0=lat0,
            lon0=lon0,
            rssi0_dbm=args.rssi0,
            n_pathloss=args.n,
        )

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # ---- interface unchanged
        out = {
            "device_id": device_id,
            "sid": sid,
            "ts": received_at or now,
            "t_unix": t_unix,
            "aps_rx": aps_rx,
            "used_aps": used,
            "est": {
                "lat": est_lat,
                "lon": est_lon,
                "method": method,
                "used": len(used)
            }
        }
        c.publish(args.topic_out, json.dumps(out, ensure_ascii=False), qos=0, retain=False)

    def run_once() -> int:
        sent = 0
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    env = json.loads(line)
                except Exception:
                    continue

                dp = (((env.get("uplink_message") or {}).get("decoded_payload")) or {})
                if not isinstance(dp, dict):
                    continue

                scan = reasm.push(dp)
                if scan is None:
                    continue

                publish_geo(env, scan)
                sent += 1

                time.sleep(args.min_sleep + (args.max_sleep - args.min_sleep) * 0.5)
        return sent

    while True:
        n = run_once()
        print(f"Published {n} geo_result messages to {args.topic_out}")
        if not args.loop:
            break
        time.sleep(1)

    c.loop_stop()
    c.disconnect()


if __name__ == "__main__":
    main()
