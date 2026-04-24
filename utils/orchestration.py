"""
High-level orchestration: dataset row → ML → optional intel → firewall → alert bus.
Use from workers or tests; keeps Streamlit/API logic thin.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.alert_bus import append_alert
from utils.behavioral_profiles import combine_scores, get_behavior_store
from utils.firewall import get_firewall_manager
from utils.prediction import AuraPredictor
def score_and_respond(
    predictor: AuraPredictor,
    records: List[Dict[str, Any]],
    *,
    use_llm: bool = False,
    auto_firewall: bool = False,
    subject_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out = predictor.predict_records(records, include_explanation=True, use_llm=use_llm)
    now = datetime.now(timezone.utc).isoformat()
    for row in out:
        key = subject_key or str(
            (row.get("live_meta") or {}).get("observed_source_ip") or row.get("cve_id") or "unknown"
        )
        store = get_behavior_store()
        dev = store.deviation_score(key, str(row.get("risk_class", "Safe")))
        store.record_event(key, str(row.get("risk_class", "Safe")), now)
        row["behavioral"] = combine_scores(int(row.get("risk_code", 0)), dev)
        append_alert({"ts": now, "cve_id": row.get("cve_id"), "risk_class": row.get("risk_class"), "source": "orchestration"})
        if auto_firewall and row.get("live_meta"):
            ip = row["live_meta"].get("observed_source_ip")
            if ip and row.get("risk_class") in ("Malicious", "Critical"):
                get_firewall_manager().block_observed_ip(str(ip), str(row.get("risk_class")), "orchestration")
    return out
