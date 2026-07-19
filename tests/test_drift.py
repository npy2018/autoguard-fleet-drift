from autoguard_drift.demo import run
from autoguard_drift.fingerprint import ContextualFingerprint
from autoguard_drift.schemas import FleetObservation


def test_demo_detects_version_shift() -> None:
    result = run()
    assert result["peak_drift_probability"] >= 0.7
    assert result["recommendation"] == "pause rollout"


def test_cold_start_falls_back_to_global() -> None:
    items = [
        FleetObservation(timestamp=i, vehicle_id="x", version="V1", metric_name="m", metric_value=1 + i * 0.01, vehicle_model="A", city="S", weather="sun", road_type="urban")
        for i in range(30)
    ]
    import pandas as pd

    fp = ContextualFingerprint().fit(pd.DataFrame([x.model_dump() for x in items]))
    stats = fp.predict({"vehicle_model": "NEW", "city": "N", "weather": "rain", "road_type": "rural"})
    assert stats.count == 30
