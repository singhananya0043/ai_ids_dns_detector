"""
IDS Detection Model
Dual-model approach:
  1. Isolation Forest  — unsupervised anomaly scoring (works without labels)
  2. Random Forest     — supervised classifier (trained on labeled data)

At inference: IF flags anomaly → RF classifies the attack type.
Auto-calibrated IF threshold using training score percentile.
Rule-based overrides for high-confidence signals.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IF_THRESHOLD_PERCENTILE = 3   # bottom N% of training scores → anomaly
IF_CONTAMINATION        = 0.1 # expected fraction of anomalies

ATTACK_LABELS = ["port_scan", "ddos", "brute_force", "lateral_movement"]
NORMAL_LABEL  = "normal"


# ─────────────────────────────────────────────
# Rule-Based Override Checks
# ─────────────────────────────────────────────

def _apply_rules(row: pd.Series) -> Optional[str]:
    """
    Return an attack label if high-confidence rule fires, else None.

    Rules:
      - port_scan  : flag_syn==1, pkt_count==1, unique_dst_ports_src > 20
      - ddos       : byte_rate_src > 500_000 AND pkt_count > 1000
      - brute_force: conn_rate_src > 5 AND unique_dst_ports_src == 1 AND pkt_count < 20
    """
    try:
        if (row.get("flag_syn", 0) == 1 and
                row.get("pkt_count", 0) <= 2 and
                row.get("unique_dst_ports_src", 0) > 20):
            return "port_scan"

        if (row.get("byte_rate_src", 0) > 500_000 and
                row.get("pkt_count", 0) > 1000):
            return "ddos"

        if (row.get("conn_rate_src", 0) > 5 and
                row.get("unique_dst_ports_src", 0) == 1 and
                row.get("pkt_count", 0) < 20):
            return "brute_force"
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# IDS Model
# ─────────────────────────────────────────────

class IDSModel:
    """
    Combined Isolation Forest + Random Forest IDS model.

    Train:
        model = IDSModel()
        model.train(feat_df_normal, feat_df_all, labels)

    Predict (returns per-row dict with score, is_anomaly, attack_type):
        results = model.predict(feat_df)
    """

    def __init__(self):
        self.if_model:   Optional[IsolationForest]     = None
        self.rf_model:   Optional[RandomForestClassifier] = None
        self.scaler:     Optional[StandardScaler]       = None
        self.le:         Optional[LabelEncoder]         = None
        self.threshold:  float                          = -0.1
        self.feature_names: List[str]                   = []

    # ── Training ─────────────────────────────────────

    def train(
        self,
        normal_feats: pd.DataFrame,
        all_feats:    pd.DataFrame,
        labels:       pd.Series,
    ) -> Dict:
        """
        Train both models.

        Args:
            normal_feats : Feature matrix of NORMAL traffic only (for IF).
            all_feats    : Feature matrix of all traffic (for RF).
            labels       : Ground-truth labels aligned with all_feats.

        Returns:
            dict with training metrics.
        """
        self.feature_names = list(all_feats.columns)

        # ── Scaler (fit on normal only, transform all) ──
        self.scaler = StandardScaler()
        X_normal_scaled = self.scaler.fit_transform(normal_feats.values)
        X_all_scaled    = self.scaler.transform(all_feats.values)

        # ── Isolation Forest ────────────────────────────
        self.if_model = IsolationForest(
            n_estimators=200,
            contamination=IF_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self.if_model.fit(X_normal_scaled)

        # Auto-calibrate threshold on training data
        train_scores = self.if_model.decision_function(X_normal_scaled)
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

        # IF recall on attack samples
        attack_mask = labels != NORMAL_LABEL
        if_tp = sum(1 for p, a in zip(if_preds, attack_mask) if p == "anomaly" and a)
        if_recall = if_tp / max(attack_mask.sum(), 1)

        rf_f1 = f1_score(labels, rf_preds, average="weighted", zero_division=0)

        return {
            "if_threshold":   round(self.threshold, 4),
            "if_attack_recall": round(if_recall, 4),
            "rf_weighted_f1": round(rf_f1, 4),
            "n_features":     len(self.feature_names),
            "classes":        list(self.le.classes_),
        }

    # ── Prediction ───────────────────────────────────

    def predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly status and attack type for each flow.

        Returns:
            DataFrame with columns:
              anomaly_score  : IF decision score (lower = more anomalous)
              is_anomaly     : bool
              attack_type    : predicted label (normal / attack class)
              confidence     : RF max probability
              rule_override  : rule that triggered (if any)
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

            # Check rule overrides
            rule = _apply_rules(feat_df.iloc[i])

            if rule:
                attack_type = rule
            elif is_anomaly:
                attack_type = rf_label if rf_label != NORMAL_LABEL else "unknown_anomaly"
            else:
                attack_type = NORMAL_LABEL

            results.append({
                "anomaly_score":  round(float(score), 5),
                "is_anomaly":     bool(is_anomaly or rule),
                "attack_type":    attack_type,
                "confidence":     round(float(conf), 4),
                "rule_override":  rule or "",
            })

        return pd.DataFrame(results)

    # ── Persistence ──────────────────────────────────

    def save(self, name: str = "ids") -> Path:
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

    def load(self, name: str = "ids") -> None:
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
    from data.simulator import NetworkFlowSimulator
    from features.ids_features import IDSFeatureExtractor, STATIC_FEATURES

    sim = NetworkFlowSimulator(seed=42)
    df  = sim.generate(n_normal=500)

    extractor = IDSFeatureExtractor(window_sec=60)
    feat_df, labels = extractor.extract_df(df)

    normal_mask = labels == NORMAL_LABEL
    normal_feats = feat_df[normal_mask].reset_index(drop=True)

    model = IDSModel()
    metrics = model.train(normal_feats, feat_df, labels)
    print("Training metrics:", metrics)

    preds = model.predict(feat_df)
    print("\nPrediction sample:\n", preds.head(10).to_string())
    print("\nAttack type distribution:\n", preds["attack_type"].value_counts())

    # Evaluate recall on true attacks
    true_attacks  = labels != NORMAL_LABEL
    pred_attacks  = preds["is_anomaly"]
    recall = (true_attacks & pred_attacks).sum() / max(true_attacks.sum(), 1)
    print(f"\nAttack recall: {recall:.2%}")

    path = model.save()
    print(f"Model saved to {path}")
