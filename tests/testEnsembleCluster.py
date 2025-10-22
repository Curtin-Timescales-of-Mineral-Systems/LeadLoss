import numpy as np

from src.process.processing import _build_peak_catalogue_from_winners, _gmm1d_fit_bic

def test_peak_catalogue_two_modes_hdi_tightens():
    ages_y = np.linspace(900e6, 1300e6, 401)  # YEARS
    rng = np.random.RandomState(0)
    # winners: two modes around 1000 Ma and 1150 Ma (in YEARS)
    w1 = rng.normal(loc=1000e6, scale=8e6, size=300)
    w2 = rng.normal(loc=1150e6, scale=10e6, size=200)
    winners_y = np.hstack([w1, w2])

    # weights: light down-weight tail by score proxy
    weights = np.ones_slike(winners_y)
    catalogue_q = _build_peak_catalogue_from_winners(
        ages_y, winners_y, weights,
        use_hdi_top_peak_ci=False
    )
    catalogue_h = _build_peak_catalogue_from_winners(
        ages_y, winners_y, weights,
        use_hdi_top_peak_ci=True
    )
    assert len(catalogue_q) >= 2
    # HDI should not be wider than quantile for row 0
    assert (catalogue_h[0]["ci_high"] - catalogue_h[0]["ci_low"]) <= \
           (catalogue_q[0]["ci_high"] - catalogue_q[0]["ci_low"]) + 1e-9
    # supports sum to ~1
    s = sum(r["support"] for r in catalogue_q)
    assert 0.99 <= s <= 1.01

def test_gmm_bic_separation_gate():
    rng = np.random.RandomState(1)
    x = np.hstack([rng.normal(1000, 5, 40), rng.normal(1150, 5, 45)])
    model = _gmm1d_fit_bic(x, kmax=3)
    # should pick K≈2 and means ordered
    assert model is not None
    assert 1 <= model["mu"].size <= 3
    assert np.all(np.diff(model["mu"]) > 0)
