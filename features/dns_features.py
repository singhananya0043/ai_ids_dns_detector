"""
DNS Feature Extractor
Converts raw DNSEvent records into ML-ready feature vectors.

Features cover:
  - Domain entropy & length (high entropy → DGA / tunneling)
  - Subdomain depth & max label length (tunneling uses deep, long subdomains)
  - Query type encoding
  - Response characteristics (NXDOMAIN rate, IP count, TTL)
  - Temporal / rate features (query frequency per src_ip)
"""

import math
import re
import string
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Feature schema
# ─────────────────────────────────────────────

STATIC_DNS_FEATURES = [
    "domain_length",
    "subdomain_depth",           # number of labels - 2 (TLD + SLD stripped)
    "max_label_length",          # longest label in FQDN
    "domain_entropy",            # Shannon entropy of full query_name
    "digit_ratio",               # fraction of digits in domain
    "consonant_ratio",           # fraction of consonants (DGA domains lack vowels)
    "hex_ratio",                 # fraction of hex chars (tunneling encodes data)
    "is_query_A",
    "is_query_AAAA",
    "is_query_TXT",
    "is_query_MX",
    "is_query_ANY",
    "is_query_other",
    "is_nxdomain",
    "response_ip_count",
    "ttl",
    "ttl_very_low",              # TTL < 10 → fast-flux / DGA
    "payload_size",
    "payload_size_large",        # > 512 bytes → possible tunneling
    "is_private_src",
]

TEMPORAL_DNS_FEATURES = [
    "query_rate_src",            # queries/sec from src_ip in 60s window
    "unique_domains_src",        # distinct domains queried by src_ip
    "nxdomain_rate_src",         # NXDOMAIN fraction for src_ip
    "avg_domain_entropy_src",    # rolling avg entropy (DGA → high, steady)
]

ALL_DNS_FEATURES = STATIC_DNS_FEATURES + TEMPORAL_DNS_FEATURES

PRIVATE_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                    "172.29.", "172.30.", "172.31.")

CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
HEX_CHARS  = set("0123456789abcdefABCDEF")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _domain_parts(fqdn: str):
    """Return (labels, sld, tld) for a FQDN."""
    labels = fqdn.rstrip(".").split(".")
    tld    = labels[-1]  if len(labels) >= 1 else ""
    sld    = labels[-2]  if len(labels) >= 2 else ""
    return labels, sld, tld


def _is_private(ip: str) -> int:
    return int(ip.startswith(PRIVATE_PREFIXES))


# ─────────────────────────────────────────────
# Stateless extractor (static features)
# ─────────────────────────────────────────────

def extract_static_dns_features(row: pd.Series) -> Dict[str, float]:
    """Extract static DNS features from one DNSEvent row."""
    qname    = str(row.get("query_name", "")).lower()
    qtype    = str(row.get("query_type", "A")).upper()
    rcode    = str(row.get("response_code", "NOERROR")).upper()
    ttl      = int(row.get("ttl", 0))
    psize    = int(row.get("payload_size", 0))
    n_ips    = len(row.get("response_ips", []))
    src_ip   = str(row.get("src_ip", ""))

    labels, sld, tld = _domain_parts(qname)
    # subdomain depth = labels beyond SLD.TLD
    subdomain_depth = max(0, len(labels) - 2)
    max_label_len   = max((len(l) for l in labels), default=0)

    # character ratios (computed on full qname, minus dots)
    chars = qname.replace(".", "")
    n_chars = max(len(chars), 1)
    digit_ratio     = sum(c.isdigit()      for c in chars) / n_chars
    consonant_ratio = sum(c in CONSONANTS  for c in chars) / n_chars
    hex_ratio       = sum(c in HEX_CHARS   for c in chars) / n_chars

    feats: Dict[str, float] = {
        "domain_length":       float(len(qname)),
        "subdomain_depth":     float(subdomain_depth),
        "max_label_length":    float(max_label_len),
        "domain_entropy":      _entropy(qname),
        "digit_ratio":         digit_ratio,
        "consonant_ratio":     consonant_ratio,
        "hex_ratio":           hex_ratio,
        "is_query_A":          int(qtype == "A"),
        "is_query_AAAA":       int(qtype == "AAAA"),
        "is_query_TXT":        int(qtype == "TXT"),
        "is_query_MX":         int(qtype == "MX"),
        "is_query_ANY":        int(qtype == "ANY"),
        "is_query_other":      int(qtype not in {"A", "AAAA", "TXT", "MX", "ANY"}),
        "is_nxdomain":         int(rcode == "NXDOMAIN"),
        "response_ip_count":   float(n_ips),
        "ttl":                 float(ttl),
        "ttl_very_low":        int(ttl < 10),
        "payload_size":        float(psize),
        "payload_size_large":  int(psize > 512),
        "is_private_src":      _is_private(src_ip),
    }
    return feats


def extract_static_dns_df(df: pd.DataFrame) -> pd.DataFrame:
    records = [extract_static_dns_features(row) for _, row in df.iterrows()]
    return pd.DataFrame(records, columns=STATIC_DNS_FEATURES)


# ─────────────────────────────────────────────
# Stateful extractor (temporal features)
# ─────────────────────────────────────────────

class DNSFeatureExtractor:
    """
    Stateful extractor: sliding-window temporal features per src_ip.

    Usage:
        extractor = DNSFeatureExtractor(window_sec=60)
        feat_dict = extractor.extract(dns_event_row)
    """

    def __init__(self, window_sec: float = 60.0):
        self.window = window_sec
        # deques store (timestamp, value) pairs
        self._query_times:  Dict[str, deque] = defaultdict(deque)
        self._domains:      Dict[str, deque] = defaultdict(deque)
        self._nxdomains:    Dict[str, deque] = defaultdict(deque)
        self._entropies:    Dict[str, deque] = defaultdict(deque)

    def _prune(self, dq: deque, now: float) -> None:
        while dq and dq[0][0] < now - self.window:
            dq.popleft()

    def extract(self, row: pd.Series) -> Dict[str, float]:
        feats   = extract_static_dns_features(row)
        ts      = float(row["timestamp"])
        src     = str(row.get("src_ip", ""))
        qname   = str(row.get("query_name", "")).lower()
        rcode   = str(row.get("response_code", "NOERROR")).upper()
        entropy = feats["domain_entropy"]

        # Prune old entries
        for dq in (self._query_times[src], self._domains[src],
                   self._nxdomains[src], self._entropies[src]):
            self._prune(dq, ts)

        # Update windows
        self._query_times[src].append((ts, 1))
        self._domains[src].append((ts, qname))
        self._nxdomains[src].append((ts, int(rcode == "NXDOMAIN")))
        self._entropies[src].append((ts, entropy))

        win = self.window
        n_queries    = len(self._query_times[src])
        uniq_domains = len({v for _, v in self._domains[src]})
        nxdomain_cnt = sum(v for _, v in self._nxdomains[src])
        nxdomain_rate = nxdomain_cnt / max(n_queries, 1)
        avg_entropy  = (sum(v for _, v in self._entropies[src])
                        / max(n_queries, 1))

        feats.update({
            "query_rate_src":         n_queries / win,
            "unique_domains_src":     float(uniq_domains),
            "nxdomain_rate_src":      nxdomain_rate,
            "avg_domain_entropy_src": avg_entropy,
        })
        return feats

    def extract_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process entire DNS event DataFrame in timestamp order.

        Returns:
            features_df : shape (n, len(ALL_DNS_FEATURES))
            labels      : ground-truth label series
        """
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        records = [self.extract(row) for _, row in df_sorted.iterrows()]
        feat_df = pd.DataFrame(records, columns=ALL_DNS_FEATURES)
        labels  = (df_sorted["label"] if "label" in df_sorted.columns
                   else pd.Series(["unknown"] * len(df_sorted)))
        return feat_df, labels


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"C:\Users\HP\ai_ids_dns_detector")
    from data.simulator import DNSSimulator

    sim    = DNSSimulator(seed=42)
    df     = sim.generate(n_normal=200)

    extractor = DNSFeatureExtractor(window_sec=60)
    feat_df, labels = extractor.extract_df(df)

    print("DNS Feature matrix shape:", feat_df.shape)
    print("Label distribution:\n", labels.value_counts())
    print("\nSample features:\n", feat_df.head(3).to_string())

    # Spot check: tunneling should have high entropy + large payload
    tunnel_mask = labels == "dns_tunneling"
    print("\nDNS Tunneling avg domain_entropy:",
          feat_df.loc[tunnel_mask.values, "domain_entropy"].mean())
    print("Normal avg domain_entropy:",
          feat_df.loc[(labels == "normal").values, "domain_entropy"].mean())
