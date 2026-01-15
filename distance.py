import json, csv, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# -----------------------------
# 你需要按情况调的参数
# -----------------------------
AP_DB_CSV = "ap_db_fake.csv"
UPLINKS_JSONL = "sim_uplinks.jsonl"

# 用于 RSSI->距离 的 log-distance 模型参数（和你生成模拟数据一致会更准）
RSSI0_DBM = -45.0    # 1m参考RSSI（你生成时用的那个）
N_PATHLOSS = 2.3     # 路损指数（你生成时用的那个）
D0_M = 1.0

# KNN 指纹库（AI法）训练集设置：用"模型"在地图上采样生成"伪训练数据"
# 注意：这是"用仿真生成训练集"，不是实采指纹；但能满足文档"IA/KNN"对比要求
TRAIN_SAMPLES = 2000
TRAIN_RADIUS_M = 100.0     # 你伪造DB的半径是100m，就用同尺度
KNN_K = 5                  # 降低K值，使用加权平均
MISSING_RSSI = -100.0      # 观测里没出现的AP，用-100填充（越小表示越弱/不可见）
RSSI_NOISE_SIGMA = 1.5     # 训练数据也要加噪声，但是和测试数据不同

# 参与向量的AP数量上限（太多维会慢）；你LoRa一般只带前6个
TOP_AP_FOR_VEC = 12

# -----------------------------
# 地理转换（小范围：经纬度 <-> 本地平面）
# -----------------------------
EARTH_R = 6378137.0

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
# RSSI <-> 距离（log-distance模型）
# -----------------------------
def distance_from_rssi(rssi_dbm: float,
                       rssi0_dbm: float = RSSI0_DBM,
                       d0_m: float = D0_M,
                       n: float = N_PATHLOSS) -> float:
    return d0_m * (10.0 ** ((rssi0_dbm - rssi_dbm) / (10.0 * n)))

def rssi_from_distance(d_m: float,
                       rssi0_dbm: float = RSSI0_DBM,
                       d0_m: float = D0_M,
                       n: float = N_PATHLOSS,
                       noise_sigma: float = 0.0) -> float:
    d = max(d_m, 0.3)
    base = rssi0_dbm - 10.0 * n * math.log10(d / d0_m)
    if noise_sigma > 0:
        return base + random.gauss(0.0, noise_sigma)
    return base

# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class AP:
    bssid: str
    lat: float
    lon: float
    source: str

@dataclass
class Obs:
    bssid: str
    rssi: float

# -----------------------------
# 读文件
# -----------------------------
def load_ap_db(path: str) -> Dict[str, AP]:
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            bssid = (row.get("bssid") or "").strip()
            if not bssid:
                continue
            out[bssid] = AP(
                bssid=bssid,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                source=(row.get("source") or "").strip(),
            )
    return out

def iter_uplinks(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def get_obs_from_uplink(u: dict) -> Tuple[str, List[Obs]]:
    decoded = u.get("uplink_message", {}).get("decoded_payload", {})
    sid = str(decoded.get("sid", ""))
    a = decoded.get("a", []) or []
    obs = []
    for item in a:
        # item = [bssid, rssi, channel, ssid]
        if isinstance(item, list) and len(item) >= 2:
            obs.append(Obs(bssid=str(item[0]).strip(), rssi=float(item[1])))
    # 按信号强度从强到弱
    obs.sort(key=lambda x: x.rssi, reverse=True)
    return sid, obs

# -----------------------------
# 方法2：RSSI加权质心（算法法）
# -----------------------------
def estimate_weighted_centroid(obs: List[Obs], ap_map: Dict[str, AP],
                               lat0: float, lon0: float) -> Optional[Tuple[float, float]]:
    xs, ys, ws = [], [], []
    for o in obs:
        ap = ap_map.get(o.bssid)
        if not ap:
            continue
        x, y = ll_to_xy_m(ap.lat, ap.lon, lat0, lon0)
        # w: RSSI越强权重越大（把dBm转线性功率，再做个平移避免极小）
        w = 10.0 ** ((o.rssi + 100.0) / 10.0)
        xs.append(x); ys.append(y); ws.append(w)

    if len(ws) < 1:
        return None
    x_hat = float(np.average(xs, weights=ws))
    y_hat = float(np.average(ys, weights=ws))
    return xy_to_ll(x_hat, y_hat, lat0, lon0)

# -----------------------------
# 方法1：多边测距（数学法：trilateration / multilateration）
# 用 RSSI->距离，再最小二乘拟合 (Gauss-Newton)
# -----------------------------
def estimate_multilateration(obs: List[Obs], ap_map: Dict[str, AP],
                             lat0: float, lon0: float,
                             max_iter: int = 30) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    pts = []
    ds = []
    for o in obs:
        ap = ap_map.get(o.bssid)
        if not ap:
            continue
        xi, yi = ll_to_xy_m(ap.lat, ap.lon, lat0, lon0)
        di = distance_from_rssi(o.rssi)
        pts.append((xi, yi))
        ds.append(di)

    if len(ds) < 3:
        return None, None

    pts = np.array(pts, dtype=float)   # (k,2)
    ds  = np.array(ds, dtype=float)    # (k,)

    # init：AP几何中心
    x = float(pts[:,0].mean())
    y = float(pts[:,1].mean())

    for _ in range(max_iter):
        dx = x - pts[:,0]
        dy = y - pts[:,1]
        ri = np.sqrt(dx*dx + dy*dy)
        ri = np.maximum(ri, 1e-6)

        f = (ri - ds).reshape(-1, 1)  # residuals
        J = np.column_stack((dx/ri, dy/ri))

        A = J.T @ J + 1e-3*np.eye(2)  # damping
        b = -J.T @ f

        delta = np.linalg.solve(A, b).flatten()
        x += float(delta[0])
        y += float(delta[1])

        if float(np.linalg.norm(delta)) < 1e-3:
            break

    # 计算RMS残差：越小越“自洽”
    dx = x - pts[:,0]
    dy = y - pts[:,1]
    ri = np.sqrt(dx*dx + dy*dy)
    rms = float(np.sqrt(np.mean((ri - ds)**2)))

    return xy_to_ll(x, y, lat0, lon0), rms

# -----------------------------
# 方法3：KNN 指纹（AI法）
# 训练集：在区域内随机采样位置，用模型生成RSSI向量（相当于“仿真指纹库”）
# 推理：观测RSSI向量与训练向量做距离，取K近邻位置均值
# -----------------------------
def build_ap_list_for_vectors(ap_map: Dict[str, AP], lat0: float, lon0: float) -> List[str]:
    # 选AP的策略：选择在训练区域内分布较好的AP（覆盖范围广的）
    # 优先选择距离中心中等距离的AP，避免太近或太远的
    items = []
    for bssid, ap in ap_map.items():
        x, y = ll_to_xy_m(ap.lat, ap.lon, lat0, lon0)
        d = math.sqrt(x*x + y*y)
        # 偏好中等距离的AP（30-80m），它们在训练区域内更均匀分布
        score = -abs(d - 50.0)  # 距离50m越近越好
        items.append((score, bssid))
    items.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in items[:TOP_AP_FOR_VEC]]

def random_point_in_radius(lat0: float, lon0: float, radius_m: float) -> Tuple[float, float]:
    r = radius_m * math.sqrt(random.random())
    ang = 2 * math.pi * random.random()
    x = r * math.cos(ang)
    y = r * math.sin(ang)
    return xy_to_ll(x, y, lat0, lon0)

def build_fingerprint_training(ap_map: Dict[str, AP], lat0: float, lon0: float,
                               ap_list: List[str],
                               n_samples: int = TRAIN_SAMPLES,
                               radius_m: float = TRAIN_RADIUS_M,
                               seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    X = np.zeros((n_samples, len(ap_list)), dtype=float)
    Y = np.zeros((n_samples, 2), dtype=float)  # xy meters

    for i in range(n_samples):
        lat, lon = random_point_in_radius(lat0, lon0, radius_m)
        x, y = ll_to_xy_m(lat, lon, lat0, lon0)
        Y[i] = [x, y]

        for j, bssid in enumerate(ap_list):
            ap = ap_map[bssid]
            ax, ay = ll_to_xy_m(ap.lat, ap.lon, lat0, lon0)
            d = math.sqrt((x-ax)**2 + (y-ay)**2)
            # 添加噪声以匹配测试数据分布
            base_rssi = rssi_from_distance(d, noise_sigma=0.0)
            X[i, j] = base_rssi + random.gauss(0.0, RSSI_NOISE_SIGMA)

    return X, Y

def obs_to_vector(obs: List[Obs], ap_list: List[str]) -> np.ndarray:
    m = {o.bssid: o.rssi for o in obs}
    v = np.array([m.get(b, MISSING_RSSI) for b in ap_list], dtype=float)
    return v

def estimate_knn(obs: List[Obs], ap_list: List[str],
                 X_train: np.ndarray, Y_train: np.ndarray,
                 lat0: float, lon0: float,
                 k: int = KNN_K) -> Optional[Tuple[float, float]]:
    if X_train.shape[0] < k:
        return None
    v = obs_to_vector(obs, ap_list)
    
    # 改进的距离计算：对RSSI数据使用加权欧氏距离
    # 对于缺失值（MISSING_RSSI），降低其权重
    diff = X_train - v.reshape(1, -1)
    # 创建权重：非缺失值权重为1，缺失值权重降低
    weights = np.ones_like(diff)
    missing_mask = (v == MISSING_RSSI) | (X_train == MISSING_RSSI)
    weights[missing_mask] = 0.1  # 缺失值权重降低
    
    # 加权欧氏距离
    d2 = np.sum(weights * diff * diff, axis=1)
    
    # 找到k个最近邻
    idx = np.argpartition(d2, k)[:k]
    distances = np.sqrt(d2[idx])
    
    # 距离加权平均（距离越小权重越大）
    # 避免除零：给最小距离一个小的epsilon
    inv_dist = 1.0 / (distances + 1e-6)
    weights_k = inv_dist / np.sum(inv_dist)
    
    xy_hat = np.average(Y_train[idx], axis=0, weights=weights_k)
    return xy_to_ll(float(xy_hat[0]), float(xy_hat[1]), lat0, lon0)

# -----------------------------
# 评估指标：把“估计点”带回去看 RSSI/距离 是否自洽
# - 对 multilateration：我们已经算了 range residual RMS (m)
# - 对 centroid / knn：用“估计点”预测每个AP的RSSI，与观测RSSI算RMSE(dB)
# -----------------------------
def rssi_rmse_at_position(est_ll: Tuple[float,float], obs: List[Obs],
                          ap_map: Dict[str, AP], lat0: float, lon0: float) -> Optional[float]:
    lat, lon = est_ll
    x, y = ll_to_xy_m(lat, lon, lat0, lon0)
    errs = []
    for o in obs:
        ap = ap_map.get(o.bssid)
        if not ap:
            continue
        ax, ay = ll_to_xy_m(ap.lat, ap.lon, lat0, lon0)
        d = math.sqrt((x-ax)**2 + (y-ay)**2)
        pred = rssi_from_distance(d)
        errs.append((pred - o.rssi)**2)
    if not errs:
        return None
    return float(math.sqrt(sum(errs)/len(errs)))

def main():
    ap_map = load_ap_db(AP_DB_CSV)
    if not ap_map:
        raise RuntimeError("AP DB is empty. Check ap_db_fake.csv")

    # 参考原点：用AP的均值（比固定中心更稳）
    lat0 = float(np.mean([ap.lat for ap in ap_map.values()]))
    lon0 = float(np.mean([ap.lon for ap in ap_map.values()]))

    # KNN训练（AI法）
    ap_list = build_ap_list_for_vectors(ap_map, lat0, lon0)
    X_train, Y_train = build_fingerprint_training(ap_map, lat0, lon0, ap_list)

    rows = []
    for u in iter_uplinks(UPLINKS_JSONL):
        sid, obs = get_obs_from_uplink(u)
        if len(obs) < 1:
            continue

        # 方法：加权质心
        est_wc = estimate_weighted_centroid(obs, ap_map, lat0, lon0)
        wc_rmse = rssi_rmse_at_position(est_wc, obs, ap_map, lat0, lon0) if est_wc else None

        # 方法：多边测距（数学法）
        est_ml, ml_rms_m = estimate_multilateration(obs, ap_map, lat0, lon0)
        ml_rmse = rssi_rmse_at_position(est_ml, obs, ap_map, lat0, lon0) if est_ml else None

        # 方法：KNN指纹（AI法）
        est_knn = estimate_knn(obs, ap_list, X_train, Y_train, lat0, lon0, k=KNN_K)
        knn_rmse = rssi_rmse_at_position(est_knn, obs, ap_map, lat0, lon0) if est_knn else None

        rows.append({
            "sid": sid,
            "n_obs": len(obs),

            "wc_lat": est_wc[0] if est_wc else None,
            "wc_lon": est_wc[1] if est_wc else None,
            "wc_rssi_rmse_db": wc_rmse,

            "ml_lat": est_ml[0] if est_ml else None,
            "ml_lon": est_ml[1] if est_ml else None,
            "ml_range_rms_m": ml_rms_m,
            "ml_rssi_rmse_db": ml_rmse,

            "knn_lat": est_knn[0] if est_knn else None,
            "knn_lon": est_knn[1] if est_knn else None,
            "knn_rssi_rmse_db": knn_rmse,
        })

    df = pd.DataFrame(rows)
    
    # 添加综合比较列
    def safe_mean(s):
        s = pd.to_numeric(s, errors="coerce")
        return float(s.dropna().mean()) if s.notna().any() else None
    
    # 为每行确定最佳方法（基于RSSI_RMSE，越小越好）
    def get_best_method(row):
        methods = []
        if pd.notna(row["wc_rssi_rmse_db"]):
            methods.append(("WC", row["wc_rssi_rmse_db"]))
        if pd.notna(row["ml_rssi_rmse_db"]):
            methods.append(("ML", row["ml_rssi_rmse_db"]))
        if pd.notna(row["knn_rssi_rmse_db"]):
            methods.append(("KNN", row["knn_rssi_rmse_db"]))
        if not methods:
            return None
        return min(methods, key=lambda x: x[1])[0]
    
    # 为每行计算排名（基于RSSI_RMSE，越小越好）
    def get_rank(row):
        methods = []
        if pd.notna(row["wc_rssi_rmse_db"]):
            methods.append(("WC", row["wc_rssi_rmse_db"]))
        if pd.notna(row["ml_rssi_rmse_db"]):
            methods.append(("ML", row["ml_rssi_rmse_db"]))
        if pd.notna(row["knn_rssi_rmse_db"]):
            methods.append(("KNN", row["knn_rssi_rmse_db"]))
        if not methods:
            return None
        # 按RSSI_RMSE排序（越小越好）
        sorted_methods = sorted(methods, key=lambda x: x[1])
        ranks = {}
        current_rank = 1
        prev_rmse = None
        for method, rmse in sorted_methods:
            # 如果与前一个值相同，使用相同排名
            if prev_rmse is not None and abs(rmse - prev_rmse) < 1e-6:
                ranks[method] = current_rank - 1
            else:
                ranks[method] = current_rank
                current_rank += 1
            prev_rmse = rmse
        return ranks
    
    df["best_method"] = df.apply(get_best_method, axis=1)
    df["wc_rank"] = df.apply(lambda row: get_rank(row).get("WC") if get_rank(row) else None, axis=1)
    df["ml_rank"] = df.apply(lambda row: get_rank(row).get("ML") if get_rank(row) else None, axis=1)
    df["knn_rank"] = df.apply(lambda row: get_rank(row).get("KNN") if get_rank(row) else None, axis=1)
    
    # 计算汇总统计
    summary_rows = []
    
    # 方法获胜次数统计
    best_counts = df["best_method"].value_counts().to_dict()
    summary_rows.append({
        "sid": "=== SUMMARY ===",
        "n_obs": None,
        "wc_lat": None, "wc_lon": None, "wc_rssi_rmse_db": None,
        "ml_lat": None, "ml_lon": None, "ml_range_rms_m": None, "ml_rssi_rmse_db": None,
        "knn_lat": None, "knn_lon": None, "knn_rssi_rmse_db": None,
        "best_method": f"Best counts: WC={best_counts.get('WC', 0)}, ML={best_counts.get('ML', 0)}, KNN={best_counts.get('KNN', 0)}",
        "wc_rank": None, "ml_rank": None, "knn_rank": None,
    })
    
    # 平均误差统计
    summary_rows.append({
        "sid": "=== AVERAGE RMSE ===",
        "n_obs": safe_mean(df["n_obs"]),
        "wc_lat": None, "wc_lon": None, "wc_rssi_rmse_db": safe_mean(df["wc_rssi_rmse_db"]),
        "ml_lat": None, "ml_lon": None, "ml_range_rms_m": safe_mean(df["ml_range_rms_m"]), 
        "ml_rssi_rmse_db": safe_mean(df["ml_rssi_rmse_db"]),
        "knn_lat": None, "knn_lon": None, "knn_rssi_rmse_db": safe_mean(df["knn_rssi_rmse_db"]),
        "best_method": None,
        "wc_rank": None, "ml_rank": None, "knn_rank": None,
    })
    
    # 平均排名统计
    summary_rows.append({
        "sid": "=== AVERAGE RANK ===",
        "n_obs": None,
        "wc_lat": None, "wc_lon": None, "wc_rssi_rmse_db": None,
        "ml_lat": None, "ml_lon": None, "ml_range_rms_m": None, "ml_rssi_rmse_db": None,
        "knn_lat": None, "knn_lon": None, "knn_rssi_rmse_db": None,
        "best_method": None,
        "wc_rank": safe_mean(df["wc_rank"]),
        "ml_rank": safe_mean(df["ml_rank"]),
        "knn_rank": safe_mean(df["knn_rank"]),
    })
    
    # 将汇总行添加到DataFrame
    df_summary = pd.DataFrame(summary_rows)
    df_with_summary = pd.concat([df, df_summary], ignore_index=True)
    
    # 保存CSV
    df_with_summary.to_csv("position_estimates_compare.csv", index=False, encoding="utf-8")
    print("✅ wrote: position_estimates_compare.csv")
    print(df.head(10))
    
    # 打印汇总信息
    print("\n=== SUMMARY ===")
    print(f"Total samples: {len(df)}")
    print(f"Average observations per sample: {safe_mean(df['n_obs']):.2f}")
    print(f"\nBest method counts:")
    for method, count in best_counts.items():
        print(f"  {method}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nAverage RSSI RMSE (dB):")
    print(f"  WC:  {safe_mean(df['wc_rssi_rmse_db']):.3f}")
    print(f"  ML:  {safe_mean(df['ml_rssi_rmse_db']):.3f}")
    print(f"  KNN: {safe_mean(df['knn_rssi_rmse_db']):.3f}")
    print(f"\nAverage Rank (1=best, lower is better):")
    print(f"  WC:  {safe_mean(df['wc_rank']):.2f}")
    print(f"  ML:  {safe_mean(df['ml_rank']):.2f}")
    print(f"  KNN: {safe_mean(df['knn_rank']):.2f}")

if __name__ == "__main__":
    main()
