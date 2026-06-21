from src.risk_engine import RISK_BANDS


def test_risk_bands_are_ordered():
    thresholds = [t for t, _, _ in RISK_BANDS]
    assert thresholds == sorted(thresholds)


def test_risk_bands_cover_full_range():
    assert RISK_BANDS[-1][0] >= 1.0
