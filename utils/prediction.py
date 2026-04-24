"""
Model loading, batch inference, and AI-style explanation strings for AURA.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from utils.helpers import MODELS_DIR, setup_logging
from utils.preprocessing import (
    RISK_NAMES,
    build_feature_frame,
    load_raw_dataset,
    transform_with_preprocessor,
)

logger = setup_logging("aura.prediction")

RISK_MODEL_PATH = MODELS_DIR / "risk_model.pkl"
CVSS_MODEL_PATH = MODELS_DIR / "cvss_model.pkl"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.h5"


@lru_cache(maxsize=8)
def _cached_joblib_load(path_str: str) -> Any:
    """Cache deserialized artifacts to avoid repeated disk IO across modules/sessions."""
    return joblib.load(Path(path_str))


@dataclass
class LoadedRiskBundle:
    classifier: Any
    isolation_forest: Any
    preprocessor: Any
    lstm_seq_len: int
    metrics: Dict[str, Any]


@dataclass
class LoadedCvssBundle:
    regressor: Any
    preprocessor: Any
    metrics: Dict[str, Any]


def _load_keras_model(path: Path):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to load lstm_model.h5. Use Python 3.10 to 3.12 with tensorflow installed."
        ) from exc
    if not path.exists():
        raise FileNotFoundError(f"LSTM model missing at {path}")
    return keras.models.load_model(path)


class AuraPredictor:
    """
    Loads persisted artifacts and scores tabular firewall/CVE records.
    Designed so a future FastAPI layer can wrap `predict_records` with thin HTTP handlers.
    """

    def __init__(
        self,
        risk_path: Path = RISK_MODEL_PATH,
        cvss_path: Path = CVSS_MODEL_PATH,
        lstm_path: Path = LSTM_MODEL_PATH,
    ) -> None:
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except Exception:
            # If sklearn isn't available yet, defer to normal import behavior.
            pass
        if not risk_path.exists():
            raise FileNotFoundError(f"risk_model.pkl not found at {risk_path}; run train_model.py first")
        if not cvss_path.exists():
            raise FileNotFoundError(f"cvss_model.pkl not found at {cvss_path}; run train_model.py first")
        risk_blob = _cached_joblib_load(str(risk_path))
        cvss_blob = _cached_joblib_load(str(cvss_path))
        self.risk = LoadedRiskBundle(
            classifier=risk_blob["classifier"],
            isolation_forest=risk_blob["isolation_forest"],
            preprocessor=risk_blob["preprocessor"],
            lstm_seq_len=int(risk_blob.get("lstm_seq_len", 12)),
            metrics=risk_blob.get("metrics", {}),
        )
        self.cvss = LoadedCvssBundle(
            regressor=cvss_blob["regressor"],
            preprocessor=cvss_blob["preprocessor"],
            metrics=cvss_blob.get("metrics", {}),
        )
        self._lstm = None
        self._lstm_path = lstm_path
        logger.info("AuraPredictor initialized")

    @property
    def lstm(self):
        if self._lstm is None:
            self._lstm = _load_keras_model(self._lstm_path)
        return self._lstm

    def _transform(self, feature_df: pd.DataFrame) -> np.ndarray:
        X = transform_with_preprocessor(self.risk.preprocessor, feature_df)
        return np.asarray(X, dtype=np.float32)

    def _transform_cvss(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Transform features using CVSS preprocessor"""
        try:
            X = transform_with_preprocessor(self.cvss.preprocessor, feature_df)
            return np.asarray(X, dtype=np.float32)
        except Exception as e:
            logger.warning(f"CVSS transform failed, falling back: {e}")
            return self._transform(feature_df)

    def _prepare_lstm_input(self, feature_df: pd.DataFrame, record_idx: int = 0) -> np.ndarray:
        """Prepare LSTM input sequence for a record"""
        try:
            if record_idx >= len(feature_df):
                record_idx = 0
            if record_idx < self.risk.lstm_seq_len - 1:
                seq_result = self.build_sequence_for_index(feature_df, record_idx)
                if seq_result is not None:
                    return seq_result
            # Fallback: create uniform padding
            X = self._transform(feature_df.iloc[[record_idx]])
            return np.repeat(X, self.risk.lstm_seq_len, axis=0)[np.newaxis, ...]
        except Exception as e:
            logger.warning(f"LSTM input preparation failed: {e}")
            X = self._transform(feature_df.iloc[[record_idx]])
            return np.repeat(X, self.risk.lstm_seq_len, axis=0)[np.newaxis, ...]

    def predict_tabular(
        self, feature_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return risk labels, risk probabilities, anomaly flags, anomaly scores, CVSS predictions."""
        X = self._transform(feature_df)
        risk_proba = self.risk.classifier.predict_proba(X)
        risk_pred = self.risk.classifier.predict(X)
        anomaly = self.risk.isolation_forest.predict(X)
        try:
            anomaly_score = self.risk.isolation_forest.score_samples(X)
        except Exception as exc:
            logger.warning("IsolationForest.score_samples failed: %s", exc)
            anomaly_score = np.zeros(len(X), dtype=np.float64)
        cvss_pred = self.cvss.regressor.predict(X)
        cvss_pred = np.clip(cvss_pred, 0.0, 10.0)
        return (
            risk_pred.astype(np.int32),
            np.asarray(risk_proba, dtype=np.float32),
            anomaly.astype(np.int32),
            np.asarray(anomaly_score, dtype=np.float64),
            cvss_pred.astype(np.float64),
        )

    def predict_lstm_sequence(self, X_seq: np.ndarray) -> np.ndarray:
        """X_seq shape (batch, seq_len, n_features)."""
        return self.lstm.predict(X_seq, verbose=0)

    def build_sequence_for_index(
        self,
        ordered_feature_df: pd.DataFrame,
        end_index: int,
    ) -> Optional[np.ndarray]:
        """Construct a single LSTM input window ending at end_index in an ordered frame."""
        seq_len = self.risk.lstm_seq_len
        if end_index < seq_len - 1 or end_index >= len(ordered_feature_df):
            return None
        start = end_index - seq_len + 1
        block = ordered_feature_df.iloc[start : end_index + 1]
        Xw = transform_with_preprocessor(self.risk.preprocessor, block)
        return np.asarray(Xw, dtype=np.float32)[np.newaxis, ...]

    def predict_records(
        self,
        records: List[Dict[str, Any]],
        include_explanation: bool = True,
        use_llm: bool = False,
    ) -> List[Dict[str, Any]]:
        """Score arbitrary record dicts matching raw CSV schema. Auto-handles errors."""
        try:
            live_metas: List[Optional[Dict[str, Any]]] = []
            clean_rows: List[Dict[str, Any]] = []
            for r in records:
                rc = dict(r)
                lm = rc.pop("_live_meta", None)
                live_metas.append(lm if isinstance(lm, dict) else None)
                clean_rows.append(rc)
            
            if not clean_rows:
                logger.warning("No clean rows for prediction")
                return []
            
            df = pd.DataFrame(clean_rows)
            feature_df = build_feature_frame(df)
            
            try:
                preds, proba, anomaly, anomaly_score, cvss_hat = self.predict_tabular(feature_df)
            except Exception as e:
                logger.error(f"predict_tabular failed: {e}, using fallback")
                # Fallback: default predictions
                preds = np.zeros(len(feature_df), dtype=np.int32)
                proba = np.zeros((len(feature_df), len(RISK_NAMES)), dtype=np.float32)
                proba[:, 0] = 1.0
                anomaly = np.ones(len(feature_df), dtype=np.int32)
                anomaly_score = np.zeros(len(feature_df), dtype=np.float64)
                cvss_hat = np.ones(len(feature_df)) * 5.0
            
            out: List[Dict[str, Any]] = []
            for i in range(len(feature_df)):
                try:
                    label = int(preds[i]) if i < len(preds) else 0
                    label = max(0, min(label, len(RISK_NAMES) - 1))
                    name = RISK_NAMES[label]
                    
                    # Dynamic CVE column detection
                    cve_value = None
                    for cve_col in ["Data", "cve_id", "cve", "CVE", "vulnerability"]:
                        if cve_col in feature_df.columns:
                            try:
                                cve_value = feature_df[cve_col].iloc[i]
                                break
                            except (KeyError, IndexError):
                                continue
                    
                    row = {
                        "cve_id": cve_value,
                        "risk_class": name,
                        "risk_code": label,
                        "risk_probabilities": {RISK_NAMES[j]: float(proba[i][j]) for j in range(min(len(RISK_NAMES), proba.shape[1]))},
                        "anomaly_score_flag": int(anomaly[i]) if i < len(anomaly) else 0,
                        "anomaly_score": float(anomaly_score[i]) if i < len(anomaly_score) else 0.0,
                        "cvss_predicted": float(np.clip(cvss_hat[i], 0.0, 10.0)) if i < len(cvss_hat) else 5.0,
                    }
                    try:
                        row["confidence"] = float(np.max(proba[i])) if i < len(proba) else 0.0
                    except Exception:
                        row["confidence"] = 0.0
                    
                    if i < len(live_metas) and live_metas[i]:
                        row["live_meta"] = live_metas[i]
                    
                    if include_explanation:
                        try:
                            base = explain_decision(
                                risk_name=name,
                                proba_row=proba[i] if i < len(proba) else np.ones(len(RISK_NAMES)) / len(RISK_NAMES),
                                anomaly_flag=int(anomaly[i]) if i < len(anomaly) else 0,
                                cvss_pred=float(np.clip(cvss_hat[i], 0.0, 10.0)) if i < len(cvss_hat) else 5.0,
                                cwe_name=str(feature_df["cwe_name"].iloc[i]) if "cwe_name" in feature_df.columns else "",
                                summary=str(feature_df["summary"].iloc[i])[:400] if "summary" in feature_df.columns else "",
                            )
                            
                            if use_llm:
                                try:
                                    from utils.llm_explainer import enhance_explanation_dict
                                    row["explanation"] = enhance_explanation_dict(
                                        base,
                                        use_llm=True,
                                        risk_name=name,
                                        proba_row=proba[i] if i < len(proba) else np.ones(len(RISK_NAMES)) / len(RISK_NAMES),
                                        anomaly_flag=int(anomaly[i]) if i < len(anomaly) else 0,
                                        cvss_pred=float(np.clip(cvss_hat[i], 0.0, 10.0)) if i < len(cvss_hat) else 5.0,
                                        cwe_name=str(feature_df["cwe_name"].iloc[i]) if "cwe_name" in feature_df.columns else "",
                                        summary=str(feature_df["summary"].iloc[i])[:400] if "summary" in feature_df.columns else "",
                                        live_meta=live_metas[i],
                                    )
                                except Exception as llm_err:
                                    logger.warning(f"LLM enhancement failed: {llm_err}, using base explanation")
                                    row["explanation"] = base
                            else:
                                row["explanation"] = base
                        except Exception as exp_err:
                            logger.warning(f"Explanation generation failed: {exp_err}")
                            row["explanation"] = {"narrative": f"Analysis: {name}"}
                    
                    out.append(row)
                except Exception as row_err:
                    logger.warning(f"Error processing row {i}: {row_err}, using safe defaults")
                    out.append({
                        "risk_class": "Safe",
                        "risk_code": 0,
                        "risk_probabilities": {RISK_NAMES[0]: 1.0},
                        "anomaly_score_flag": 0,
                        "anomaly_score": 0.0,
                        "cvss_predicted": 0.0,
                        "confidence": 1.0,
                        "error": str(row_err)
                    })
            
            return out
        except Exception as e:
            logger.error(f"Critical error in predict_records: {e}")
            return [{"error": str(e), "risk_class": "Safe"}]


def explain_decision(
    risk_name: str,
    proba_row: np.ndarray,
    anomaly_flag: int,
    cvss_pred: float,
    cwe_name: str,
    summary: str,
) -> Dict[str, str]:
    """Turn model outputs into human-readable narrative fields."""
    top_idx = int(np.argmax(proba_row))
    top_p = float(proba_row[top_idx])
    severity_line = (
        f"Model-estimated severity (CVSS-like scale): {cvss_pred:.2f} on a 0 to 10 range. "
        f"Primary risk label: {risk_name} (confidence {top_p:.2f})."
    )
    anomaly_line = (
        "Isolation Forest flagged this profile as anomalous relative to learned normal traffic geometry."
        if anomaly_flag < 0
        else "Isolation Forest: profile consistent with bulk historical behavior (not flagged as outlier)."
    )
    cwe_line = f"Mapped weakness family: {cwe_name.strip()}" if cwe_name.strip() else "CWE metadata unavailable."
    summary_line = f"Evidence excerpt: {summary[:280]}{'...' if len(summary) > 280 else ''}"
    actions = []
    if risk_name in ("Malicious", "Critical"):
        actions.append("Block or quarantine correlated ingress until analyst review completes.")
        actions.append("Escalate to SOC with full packet metadata and identity context.")
    elif risk_name == "Suspicious":
        actions.append("Apply rate limits and enhanced logging; require step-up authentication.")
    else:
        actions.append("Allow with baseline monitoring; retain samples for drift tracking.")
    if anomaly_flag < 0:
        actions.append("Treat as high-priority anomaly: run secondary rules and threat-intel correlation.")
    return {
        "threat_summary": severity_line,
        "anomaly_context": anomaly_line,
        "weakness_context": cwe_line,
        "evidence": summary_line,
        "recommended_actions": " ".join(actions),
        "risk_severity_interpretation": interpret_risk_band(risk_name, cvss_pred),
    }


def interpret_risk_band(risk_name: str, cvss_pred: float) -> str:
    """Narrative interpretation of severity band."""
    if risk_name == "Critical" or cvss_pred >= 9.0:
        return "Critical: immediate exploitation potential; assume active targeting until disproven."
    if risk_name == "Malicious" or cvss_pred >= 6.0:
        return "Malicious: material compromise impact likely under common configurations."
    if risk_name == "Suspicious" or cvss_pred >= 4.0:
        return "Suspicious: latent weakness or abuse path; monitor and validate configuration posture."
    return "Safe: low modeled impact under current feature snapshot; maintain standard controls."


def load_ordered_feature_frame(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load dataset and return feature frame ordered by publication time (for LSTM windows)."""
    raw = load_raw_dataset(str(csv_path) if csv_path else None)
    feat = build_feature_frame(raw)
    return feat.sort_values("pub_ts").reset_index(drop=True)
