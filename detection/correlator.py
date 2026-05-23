"""
Cross-Engine Correlation Module
Fuses IDS alerts and DNS alerts from the same source IP into unified
ThreatEvent records with a combined severity score.

Correlation logic:
  - Host appears in BOTH IDS + DNS alerts → HIGH confidence, escalated severity
  - IDS attack + DGA/tunneling DNS from same IP → likely C2/exfiltration
  - Brute-force IDS + typosquatting DNS  → credential phishing campaign
  - Port-scan IDS + DGA DNS              → malware reconnaissance
  - DNS only / IDS only                  → MEDIUM severity
  - No alerts                            → clean
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────
# Severity & Event Types
# ─────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"


SEVERITY_SCORE: Dict[Severity, int] = {
    Severity.CRITICAL: 5,
    Severity.HIGH:     4,
    Severity.MEDIUM:   3,
    Severity.LOW:      2,
    Severity.INFO:     1,
}


@dataclass
class ThreatEvent:
    """Unified alert combining IDS + DNS evidence."""
    timestamp:      float
    src_ip:         str
    severity:       Severity
    threat_score:   float           # 0-10 composite score
    ids_attack:     str             # IDS attack type (or "none")
    dns_attack:     str             # DNS attack type (or "none")
    ids_confidence: float
    dns_confidence: float
    correlation:    str             # human-readable correlation reason
    raw_ids_row:    Optional[int]   # row index in IDS prediction df
    raw_dns_row:    Optional[int]   # row index in DNS prediction df


# ─────────────────────────────────────────────
# Correlation Rules
# ─────────────────────────────────────────────

# (ids_attack, dns_attack) → (severity, correlation_message)
CORRELATION_RULES: List[Tuple[set, set, Severity, str]] = [
    # Definitive C2 exfiltration
    ({"port_scan", "lateral_movement"}, {"dns_tunneling"},
     Severity.CRITICAL, "Host performing recon AND DNS exfiltration — active C2 channel"),

    ({"brute_force", "lateral_movement"}, {"dga"},
     Severity.CRITICAL, "Credential attack + DGA beaconing — likely compromised host"),

    ({"ddos"}, {"dns_amplification"},
     Severity.CRITICAL, "DDoS + DNS amplification — coordinated volumetric attack"),

    # High confidence
    ({"port_scan"}, {"dga"},
     Severity.HIGH, "Port scan + DGA queries — malware reconnaissance phase"),

    ({"brute_force"}, {"typosquatting"},
     Severity.HIGH, "Brute force + typosquatting — credential phishing campaign"),

    ({"lateral_movement"}, {"fast_flux"},
     Severity.HIGH, "Lateral movement + fast-flux DNS — botnet pivoting"),

    ({"ddos"}, {"fast_flux"},
     Severity.HIGH, "DDoS traffic + fast-flux DNS — botnet infrastructure"),

    # Any IDS + any DNS anomaly
    ({"port_scan", "ddos", "brute_force", "lateral_movement"},
     {"dns_tunneling", "dga", "typosquatting", "fast_flux", "dns_amplification"},
     Severity.HIGH, "Network intrusion attempt correlated with DNS anomaly"),
]


def _match_rule(ids_type: str, dns_type: str) -> Tuple[Severity, str]:
    """Return the best-matching severity & reason for (ids, dns) pair."""
    for ids_set, dns_set, sev, msg in CORRELATION_RULES:
        if ids_type in ids_set and dns_type in dns_set:
            return sev, msg
    # Fallback
    if ids_type != "normal" and ids_type != "none":
        return Severity.MEDIUM, f"IDS anomaly detected: {ids_type}"
    if dns_type != "normal" and dns_type != "none":
        return Severity.MEDIUM, f"DNS anomaly detected: {dns_type}"
    return Severity.INFO, "Normal traffic"


def _threat_score(
    ids_attack:   str,
    dns_attack:   str,
    ids_conf:     float,
    dns_conf:     float,
    severity:     Severity,
) -> float:
    """Compute composite threat score 0-10."""
    base   = SEVERITY_SCORE[severity] * 2          # 2-10
    bonus  = 0.0

    if ids_attack not in ("normal", "none"):
        bonus += ids_conf * 1.5
    if dns_attack not in ("normal", "none"):
        bonus += dns_conf * 1.5

    # Extra weight for cross-layer correlation
    if ids_attack not in ("normal", "none") and dns_attack not in ("normal", "none"):
        bonus += 2.0

    return min(round(base + bonus, 2), 10.0)


# ─────────────────────────────────────────────
# Correlator
# ─────────────────────────────────────────────

class ThreatCorrelator:
    """
    Correlates IDS predictions and DNS predictions on matching src_ip.

    Usage:
        correlator = ThreatCorrelator()
        events = correlator.correlate(
            flow_df,    ids_preds,
            dns_df,     dns_preds,
        )
    """

    def correlate(
        self,
        flow_df:   pd.DataFrame,
        ids_preds: pd.DataFrame,
        dns_df:    pd.DataFrame,
        dns_preds: pd.DataFrame,
    ) -> List[ThreatEvent]:
        """
        Match IDS alerts and DNS alerts by src_ip and produce ThreatEvents.

        Args:
            flow_df   : original flow DataFrame (needs 'src_ip', 'timestamp')
            ids_preds : IDSModel.predict() output (needs 'is_anomaly','attack_type','confidence')
            dns_df    : original DNS event DataFrame (needs 'src_ip', 'timestamp')
            dns_preds : DNSModel.predict() output

        Returns:
            List[ThreatEvent] sorted by threat_score descending.
        """
        events: List[ThreatEvent] = []

        # ── Index IDS alerts by src_ip ────────────────
        ids_alerts: Dict[str, List[dict]] = defaultdict(list)
        for idx, (_, flow_row) in enumerate(flow_df.iterrows()):
            pred = ids_preds.iloc[idx]
            ids_alerts[str(flow_row["src_ip"])].append({
                "ts":         float(flow_row["timestamp"]),
                "attack":     str(pred["attack_type"]),
                "confidence": float(pred["confidence"]),
                "row_idx":    idx,
                "is_anomaly": bool(pred["is_anomaly"]),
            })

        # ── Index DNS alerts by src_ip ────────────────
        dns_alerts: Dict[str, List[dict]] = defaultdict(list)
        for idx, (_, dns_row) in enumerate(dns_df.iterrows()):
            pred = dns_preds.iloc[idx]
            dns_alerts[str(dns_row["src_ip"])].append({
                "ts":         float(dns_row["timestamp"]),
                "attack":     str(pred["attack_type"]),
                "confidence": float(pred["confidence"]),
                "row_idx":    idx,
                "is_anomaly": bool(pred["is_anomaly"]),
            })

        # ── All unique IPs across both engines ────────
        all_ips = set(ids_alerts.keys()) | set(dns_alerts.keys())

        for ip in all_ips:
            ids_list = ids_alerts.get(ip, [])
            dns_list = dns_alerts.get(ip, [])

            # Pick worst (most severe) alert per engine
            ids_anomalies = [a for a in ids_list if a["is_anomaly"]]
            dns_anomalies = [a for a in dns_list if a["is_anomaly"]]

            # Best IDS attack for this IP
            if ids_anomalies:
                best_ids = max(ids_anomalies, key=lambda x: x["confidence"])
                ids_attack = best_ids["attack"]
                ids_conf   = best_ids["confidence"]
                ids_ts     = best_ids["ts"]
                ids_row    = best_ids["row_idx"]
            else:
                ids_attack, ids_conf, ids_ts, ids_row = "none", 0.0, 0.0, None

            # Best DNS attack for this IP
            if dns_anomalies:
                best_dns = max(dns_anomalies, key=lambda x: x["confidence"])
                dns_attack = best_dns["attack"]
                dns_conf   = best_dns["confidence"]
                dns_ts     = best_dns["ts"]
                dns_row    = best_dns["row_idx"]
            else:
                dns_attack, dns_conf, dns_ts, dns_row = "none", 0.0, 0.0, None

            # Skip fully clean IPs
            if ids_attack == "none" and dns_attack == "none":
                continue

            severity, reason = _match_rule(ids_attack, dns_attack)
            score = _threat_score(ids_attack, dns_attack, ids_conf, dns_conf, severity)
            ts    = max(ids_ts, dns_ts)

            events.append(ThreatEvent(
                timestamp      = ts,
                src_ip         = ip,
                severity       = severity,
                threat_score   = score,
                ids_attack     = ids_attack,
                dns_attack     = dns_attack,
                ids_confidence = round(ids_conf, 4),
                dns_confidence = round(dns_conf, 4),
                correlation    = reason,
                raw_ids_row    = ids_row,
                raw_dns_row    = dns_row,
            ))

        events.sort(key=lambda e: e.threat_score, reverse=True)
        return events

    @staticmethod
    def to_dataframe(events: List[ThreatEvent]) -> pd.DataFrame:
        """Convert ThreatEvent list to a flat DataFrame."""
        return pd.DataFrame([{
            "timestamp":      e.timestamp,
            "src_ip":         e.src_ip,
            "severity":       e.severity.value,
            "threat_score":   e.threat_score,
            "ids_attack":     e.ids_attack,
            "dns_attack":     e.dns_attack,
            "ids_confidence": e.ids_confidence,
            "dns_confidence": e.dns_confidence,
            "correlation":    e.correlation,
        } for e in events])


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"C:\Users\HP\ai_ids_dns_detector")

    from data.simulator       import NetworkFlowSimulator, DNSSimulator
    from features.ids_features import IDSFeatureExtractor
    from features.dns_features import DNSFeatureExtractor
    from detection.ids_model   import IDSModel
    from detection.dns_model   import DNSModel

    # ── Generate data ──
    net_sim = NetworkFlowSimulator(seed=7)
    dns_sim = DNSSimulator(seed=7)
    flow_df = net_sim.generate(n_normal=300)
    dns_df  = dns_sim.generate(n_normal=300)

    # ── Extract features ──
    ids_ext = IDSFeatureExtractor(window_sec=60)
    dns_ext = DNSFeatureExtractor(window_sec=60)
    ids_feat, ids_labels = ids_ext.extract_df(flow_df)
    dns_feat, dns_labels = dns_ext.extract_df(dns_df)

    # ── Train models ──
    ids_model = IDSModel()
    dns_model = DNSModel()
    ids_model.train(ids_feat[ids_labels == "normal"], ids_feat, ids_labels)
    dns_model.train(dns_feat[dns_labels == "normal"], dns_feat, dns_labels)

    # ── Predict ──
    ids_preds = ids_model.predict(ids_feat)
    dns_preds = dns_model.predict(dns_feat)

    # ── Correlate ──
    correlator = ThreatCorrelator()
    events = correlator.correlate(
        flow_df.sort_values("timestamp").reset_index(drop=True), ids_preds,
        dns_df.sort_values("timestamp").reset_index(drop=True),  dns_preds,
    )

    df = ThreatCorrelator.to_dataframe(events)
    print(f"Total threat events: {len(events)}")
    print("\nTop 10 threats:\n", df.head(10).to_string())
    print("\nSeverity distribution:\n", df["severity"].value_counts())
