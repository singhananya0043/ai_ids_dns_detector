"""
IDS Feature Extractor
Converts raw NetworkFlow records into ML-ready feature vectors.

Static features (no temporal state) → safe for Isolation Forest training.
Temporal features (rate-based, burst detection) → used only at inference.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# Feature Schema
# ─────────────────────────────────────────────

STATIC_FEATURES = [
    "pkt_count",
    "byte_count",
    "duration",
    "bytes_per_pkt",
    "pkts_per_sec",
    "bytes_per_sec",
    "dst_port",
    "src_port",
    "is_well_known_port",
    "is_tcp",
    "is_udp",
    "is_icmp",
    "flag_syn",
    "flag_ack",
    "flag_fin",
    "flag_rst",
    "flag_psh",
    "is_private_src",
    "is_private_dst",
]

TEMPORAL_FEATURES = [
    "conn_rate_src",        # connections per second from src_ip (60s window)
    "unique_dst_ports_src", # distinct dst ports touched by src_ip (60s window)
    "unique_dst_ips_src",   # distinct dst IPs from src_ip (60s window)
    "byte_rate_src",        # bytes/sec from src_ip (60s window)
]

ALL_FEATURES = STATIC_FEATURES + TEMPORAL_FEATURES

PRIVATE_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                    "172.29.", "172.30.", "172.31.")

WELL_KNOWN_PORTS = {21, 22, 23, 25, 53, 80, 110, 143, 443, 445,
                    3306, 3389, 5900, 8080, 8443}


def _is_private(ip: str) -> int:
    return int(ip.startswith(PRIVATE_PREFIXES))


def _parse_flags(flags: str) -> Dict[str, int]:
    f = flags.upper()
    return {
        "flag_syn": int("SYN" in f),
        "flag_ack": int("ACK" in f),
        "flag_fin": int("FIN" in f),
        "flag_rst": int("RST" in f),
        "flag_psh": int("PSH" in f),
    }


# ─────────────────────────────────────────────
# Stateless extractor (static features only)
# ─────────────────────────────────────────────

def extract_static_features(row: pd.Series) -> Dict[str, float]:
    """Extract static, non-temporal features from a single flow row."""
    dur = max(float(row["duration"]), 1e-6)
    pkts = max(int(row["pkt_count"]), 1)
    byts = max(int(row["byte_count"]), 1)
    proto = str(row.get("protocol", "TCP")).upper()

    feats: Dict[str, float] = {
        "pkt_count":          float(pkts),
        "byte_count":         float(byts),
        "duration":           dur,
        "bytes_per_pkt":      byts / pkts,
        "pkts_per_sec":       pkts / dur,
        "bytes_per_sec":      byts / dur,
        "dst_port":           float(row.get("dst_port", 0)),
        "src_port":           float(row.get("src_port", 0)),
        "is_well_known_port": int(int(row.get("dst_port", 0)) in WELL_KNOWN_PORTS),
        "is_tcp":             int(proto == "TCP"),
        "is_udp":             int(proto == "UDP"),
        "is_icmp":            int(proto == "ICMP"),
        "is_private_src":     _is_private(str(row.get("src_ip", ""))),
        "is_private_dst":     _is_private(str(row.get("dst_ip", ""))),
    }
    feats.update(_parse_flags(str(row.get("flags", ""))))
    return feats


def extract_static_df(df: pd.DataFrame) -> pd.DataFrame:
    """Batch-extract static features from a flow DataFrame."""
    records = [extract_static_features(row) for _, row in df.iterrows()]
    return pd.DataFrame(records, columns=STATIC_FEATURES)


# ─────────────────────────────────────────────
# Stateful extractor (temporal features)
# ─────────────────────────────────────────────

class IDSFeatureExtractor:
    """
    Stateful extractor that computes sliding-window temporal features
    in addition to static features.

    Usage:
        extractor = IDSFeatureExtractor(window_sec=60)
        feature_row = extractor.extract(flow_row)
    """

    def __init__(self, window_sec: float = 60.0):
        self.window = window_sec
        # Per-src_ip sliding windows
        self._conn_times:  Dict[str, deque] = defaultdict(deque)
        self._dst_ports:   Dict[str, deque] = defaultdict(deque)
        self._dst_ips:     Dict[str, deque] = defaultdict(deque)
        self._bytes:       Dict[str, deque] = defaultdict(deque)

    def _prune(self, dq: deque, now: float) -> None:
        """Remove entries older than the window."""
        while dq and dq[0][0] < now - self.window:
            dq.popleft()

    def extract(self, row: pd.Series) -> Dict[str, float]:
        """Return full feature dict (static + temporal) for one flow."""
        feats = extract_static_features(row)

        ts  = float(row["timestamp"])
        src = str(row.get("src_ip", ""))
        dst_port = int(row.get("dst_port", 0))
        dst_ip   = str(row.get("dst_ip", ""))
        byts     = float(row.get("byte_count", 0))

        # Update windows
        for dq in (self._conn_times[src], self._dst_ports[src],
                   self._dst_ips[src], self._bytes[src]):
            self._prune(dq, ts)

        self._conn_times[src].append((ts, 1))
        self._dst_ports[src].append((ts, dst_port))
        self._dst_ips[src].append((ts, dst_ip))
        self._bytes[src].append((ts, byts))

        win = self.window
        conn_rate   = len(self._conn_times[src]) / win
        uniq_ports  = len({v for _, v in self._dst_ports[src]})
        uniq_ips    = len({v for _, v in self._dst_ips[src]})
        byte_rate   = sum(v for _, v in self._bytes[src]) / win

        feats.update({
            "conn_rate_src":        conn_rate,
            "unique_dst_ports_src": float(uniq_ports),
            "unique_dst_ips_src":   float(uniq_ips),
            "byte_rate_src":        byte_rate,
        })
        return feats

    def extract_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process an entire DataFrame in timestamp order.

        Returns:
            features_df : DataFrame of shape (n, len(ALL_FEATURES))
            labels      : Series of ground-truth labels
        """
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        records = []
        for _, row in df_sorted.iterrows():
            records.append(self.extract(row))
        feat_df = pd.DataFrame(records, columns=ALL_FEATURES)
        labels = df_sorted["label"] if "label" in df_sorted.columns else pd.Series(["unknown"] * len(df_sorted))
        return feat_df, labels


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"C:\Users\HP\ai_ids_dns_detector")
    from data.simulator import NetworkFlowSimulator

    sim = NetworkFlowSimulator(seed=42)
    df  = sim.generate(n_normal=200)

    extractor = IDSFeatureExtractor(window_sec=60)
    feat_df, labels = extractor.extract_df(df)

    print("Feature matrix shape:", feat_df.shape)
    print("Label distribution:\n", labels.value_counts())
    print("\nSample features:\n", feat_df.head(3).to_string())
