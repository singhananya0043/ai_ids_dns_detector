"""
DNS Anomaly Detection Model
Dual-model approach:
  1. Isolation Forest  — unsupervised anomaly scorer
  2. Random Forest     — supervised attack-type classifier

Rule-based overrides for clear-cut signals:
  - Very high entropy + large payload  → dns_tunneling
  - Very low TTL + many response IPs   → fast_flux
  - Query type ANY + large payload     → dns_amplification
  - NXDOMAIN rate > 0.7 (per src_ip)  → dga
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IF_THRESHOLD_PERCENTILE = 3
NORMAL_LABEL = "normal"

DNS_ATTACK_LABELS = [
    "dns_tunneling",
    "dga",
    "typosquatting",
    "fast_flux",
    "dns_amplification",
]


# ─────────────────────────────────────────────
# Rule-Based Overrides
# ─────────────────────────────────────────────

def _apply_dns_rules(row: pd.Series) -> Optional[str]:
    """
    Return attack label if high-confidence rule fires, else None.
    """
    try:
        entropy      = row.get("domain_entropy", 0)
        psize        = row.get("payload_size", 0)
        ttl          = row.get("ttl", 300)
        n_ips        = row.get("response_ip_count", 0)
        qtype_any    = row.get("is_query_ANY", 0)
        nxd_rate     = row.get("nxdomain_rate_src", 0)
        subdepth     = row.get("subdomain_depth", 0)
        max_lbl_len  = row.get("max_label_length", 0)
        avg_ent      = row.get("avg_domain_entropy_src", 0)

        # DNS Tunneling: very deep subdomains with encoded data
        if subdepth >= 3 and max_lbl_len >= 30 and psize > 400:
            return "dns_tunneling"

        # DNS Tunneling: high entropy + large payload
        if entropy > 4.5 and psize > 512:
            return "dns_tunneling"

        # Fast-Flux: very low TTL + many IPs
        if ttl < 10 and n_ips >= 4:
            return "fast_flux"

        # DNS Amplification: ANY query with massive response
        if qtype_any == 1 and psize > 1500:
            return "dns_amplification"

        # DGA: sustained high NXDOMAIN rate from this IP
        if nxd_rate > 0.65 and avg_ent > 3.8:
            return "dga"

    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# DNS Model
# ─────────────────────────────────────────────

class DNSModel:
    """
    Combined Isolation Forest + Random Forest DNS anomaly model.

    Usage:
        model = DNSModel()
        metrics = model.train(normal_feats, all_feats, labels)
        preds   = model.predict(feat_df)
    """

    def __init__(self):
        self.if_model:      Optional[IsolationForest]        = None
        self.rf_model:      Optional[RandomForestClassifier] = None
        self.scaler:        Optional[StandardScaler]          = None
        self.le:            Optional[LabelEncoder]            = None
        self.threshold:     float                             = -0.1
        self.feature_names: List[str]                         = []

    # ── Training ─────────────────────────────────────

    def train(
        self,
        normal_feats: pd.DataFrame,
        all_feats:    pd.DataFrame,
        labels:       pd.Series,
    ) -> Dict:
        self.feature_names = list(all_feats.columns)

        self.scaler = StandardScaler()
        X_normal_scaled = self.scaler.fit_transform(normal_feats.values)
        X_all_scaled    = self.scaler.transform(all_feats.values)

        # ── Isolation Forest ────────────────────────────
        self.if_model = IsolationForest(
            n_estimators=200,
            contamination=0.12,
            random_state=42,
            n_jobs=-1,
        )
        self.if_model.fit(X_normal_scaled)

        train_scores   = self.if_model.decision_function(X_normal_scaled)
        self.threshold = float(np.percentile(train_scores, IF_THRESHOLD_PERCENTILE))

        # ── Random Forest ───────────────────────────────
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(labels)

        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_all_scaled, y_encoded)

        # ── Metrics ─────────────────────────────────────
        rf_preds  = self.le.inverse_transform(self.rf_model.predict(X_all_scaled))
        if_scores = self.if_model.decision_function(X_all_scaled)
        if_preds  = ["normal" if s >= self.threshold else "anomaly"
                     for s in if_scores]

        attack_mask  = labels != NORMAL_LABEL
        if_tp        = sum(1 for p, a in zip(if_preds, attack_mask) if p == "anomaly" and a)
        if_recall    = if_tp / max(attack_mask.sum(), 1)
        rf_f1        = f1_score(labels, rf_preds, average="weighted", zero_division=0)

        return {
            "if_threshold":     round(self.threshold, 4),
            "if_attack_recall": round(if_recall, 4),
            "rf_weighted_f1":   round(rf_f1, 4),
            "n_features":       len(self.feature_names),
            "classes":          list(self.le.classes_),
        }

    # ── Prediction ───────────────────────────────────

    def predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict DNS anomaly status and attack type.

        Returns DataFrame with columns:
          anomaly_score, is_anomaly, attack_type, confidence, rule_override
        """
        if self.if_model is None or self.rf_model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        X = self.scaler.transform(feat_df[self.feature_names].values)

        if_scores   = self.if_model.decision_function(X)
        rf_proba    = self.rf_model.predict_proba(X)
        rf_pred_idx = np.argmax(rf_proba, axis=1)
        rf_labels   = self.le.inverse_transform(rf_pred_idx)
        rf_conf     = rf_proba.max(axis=1)

        results = []
        for i, (score, rf_label, conf) in enumerate(zip(if_scores, rf_labels, rf_conf)):
            is_anomaly = score < self.threshold
            rule = _apply_dns_rules(feat_df.iloc[i])

            if rule:
                attack_type = rule
            elif is_anomaly:
                attack_type = rf_label if rf_label != NORMAL_LABEL else "unknown_dns_anomaly"
            else:
                attack_type = NORMAL_LABEL

            results.append({
                "anomaly_score": round(float(score), 5),
                "is_anomaly":    bool(is_anomaly or rule),
                "attack_type":   attack_type,
                "confidence":    round(float(conf), 4),
                "rule_override": rule or "",
            })

        return pd.DataFrame(results)

    # ── Persistence ──────────────────────────────────

    def save(self, name: str = "dns") -> Path:
        path = MODEL_DIR / f"{name}_model.pkl"
        joblib.dump({
            "if_model":      self.if_model,
            "rf_model":      self.rf_model,
            "scaler":        self.scaler,
            "le":            self.le,
            "threshold":     self.threshold,
            "feature_names": self.feature_names,
        }, path)
        return path

    def load(self, name: str = "dns") -> None:
        path = MODEL_DIR / f"{name}_model.pkl"
        data = joblib.load(path)
        self.if_model      = data["if_model"]
        self.rf_model      = data["rf_model"]
        self.scaler        = data["scaler"]
        self.le            = data["le"]
        self.threshold     = data["threshold"]
        self.feature_names = data["feature_names"]


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"C:\Users\HP\ai_ids_dns_detector")
    from data.simulator import DNSSimulator
    from features.dns_features import DNSFeatureExtractor

    sim = DNSSimulator(seed=42)
    df  = sim.generate(n_normal=500)

    extractor = DNSFeatureExtractor(window_sec=60)
    feat_df, labels = extractor.extract_df(df)

    normal_mask  = labels == NORMAL_LABEL
    normal_feats = feat_df[normal_mask].reset_index(drop=True)

    model   = DNSModel()
    metrics = model.train(normal_feats, feat_df, labels)
    print("Training metrics:", metrics)

    preds = model.predict(feat_df)
    print("\nAttack type distribution:\n", preds["attack_type"].value_counts())

    attack_mask  = labels != NORMAL_LABEL
    pred_attacks = preds["is_anomaly"]
    recall = (attack_mask.values & pred_attacks.values).sum() / max(attack_mask.sum(), 1)
    print(f"\nDNS Attack recall: {recall:.2%}")

    path = model.save()
    print(f"Model saved to {path}")
