import math
import numpy as np
from scipy.stats import ks_2samp

from process import calculations
from model.monteCarloRun import MonteCarloRun
from model.settings.calculation import ConcordiaMode


class _KSDissimilarityTest:
    """Mimic your DissimilarityTest interface using scipy's KS test."""
    def perform(self, a, b):
        res = ks_2samp(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)

    def getComparisonValue(self, test_statistics):
        # Your pipeline treats this as "base dissimilarity" (lower is better)
        return float(test_statistics[0])  # KS D


class _Settings:
    def __init__(self):
        self.concordiaMode = ConcordiaMode.WETHERILL


def _weth_to_tw(x_207_235, y_206_238):
    """Convert Wetherill (x=207/235, y=206/238) to TW (u=238/206, v=207/206)."""
    U = float(calculations.U238U235_RATIO)
    y = float(y_206_238)
    x = float(x_207_235)
    if (not math.isfinite(x)) or (not math.isfinite(y)) or y <= 0.0 or U <= 0.0:
        return (math.nan, math.nan)
    u = 1.0 / y
    v = x / (U * y)
    return u, v


def test_discordant_age_wetherill_hits_upper_intercept():
    t_lo = 1500e6
    t_hi = 3000e6

    x_lo = calculations.pb207u235_from_age(t_lo)
    y_lo = calculations.pb206u238_from_age(t_lo)
    x_hi = calculations.pb207u235_from_age(t_hi)
    y_hi = calculations.pb206u238_from_age(t_hi)

    # point on Wetherill chord
    f = 0.55
    x2 = x_hi + (x_lo - x_hi) * f
    y2 = y_hi + (y_lo - y_hi) * f

    ui = calculations.discordant_age_wetherill(x_lo, y_lo, x2, y2)
    assert ui is not None and math.isfinite(ui)
    assert abs(ui - t_hi) < 1e5  # 0.1 Ma


def test_montecarlo_run_wetherill_path_produces_finite_reconstructed_ages():
    settings = _Settings()
    test = _KSDissimilarityTest()

    t_lo = 1500e6
    t_hi = 3200e6

    # Concordant points should represent the crystallisation/upper age population
    conc_ages = np.array([3150e6, 3200e6, 3250e6], float)
    conc_xy = [(calculations.pb207u235_from_age(t), calculations.pb206u238_from_age(t)) for t in conc_ages]
    conc_u, conc_v = zip(*[_weth_to_tw(x, y) for x, y in conc_xy])
    conc_u = np.asarray(conc_u, float)
    conc_v = np.asarray(conc_v, float)

    # Discordant points: straight chord in Wetherill between (t_lo) and (t_hi)
    x_lo = calculations.pb207u235_from_age(t_lo)
    y_lo = calculations.pb206u238_from_age(t_lo)
    x_hi = calculations.pb207u235_from_age(t_hi)
    y_hi = calculations.pb206u238_from_age(t_hi)

    fracs = np.array([0.2, 0.4, 0.6, 0.8], float)
    disc_x = x_hi + (x_lo - x_hi) * fracs
    disc_y = y_hi + (y_lo - y_hi) * fracs
    disc_u, disc_v = zip(*[_weth_to_tw(x, y) for x, y in zip(disc_x, disc_y)])
    disc_u = np.asarray(disc_u, float)
    disc_v = np.asarray(disc_v, float)

    run = MonteCarloRun(
        run_number=0,
        sample_name="unit_test",
        concordant_uPb=conc_u,
        concordant_pbPb=conc_v,
        discordant_uPb=disc_u,
        discordant_pbPb=disc_v,
        discordant_labels=None,
        settings=settings,
    )

    run.samplePbLossAge(t_lo, test, penalise_invalid_ages=True)

    st = run.statistics_by_pb_loss_age[t_lo]
    assert len(st.valid_discordant_ages) > 0
    assert all(math.isfinite(a) for a in st.valid_discordant_ages)
    assert min(st.valid_discordant_ages) > t_lo


def test_wetherill_score_curve_is_not_flat_and_min_near_true_pb_loss_age():
    settings = _Settings()
    test = _KSDissimilarityTest()

    t_lo_true = 1500e6
    t_hi = 3200e6

    conc_ages = np.array([3150e6, 3200e6, 3250e6], float)
    conc_xy = [(calculations.pb207u235_from_age(t), calculations.pb206u238_from_age(t)) for t in conc_ages]
    conc_u, conc_v = zip(*[_weth_to_tw(x, y) for x, y in conc_xy])
    conc_u = np.asarray(conc_u, float)
    conc_v = np.asarray(conc_v, float)

    x_lo = calculations.pb207u235_from_age(t_lo_true)
    y_lo = calculations.pb206u238_from_age(t_lo_true)
    x_hi = calculations.pb207u235_from_age(t_hi)
    y_hi = calculations.pb206u238_from_age(t_hi)

    fracs = np.array([0.25, 0.5, 0.75], float)
    disc_x = x_hi + (x_lo - x_hi) * fracs
    disc_y = y_hi + (y_lo - y_hi) * fracs
    disc_u, disc_v = zip(*[_weth_to_tw(x, y) for x, y in zip(disc_x, disc_y)])
    disc_u = np.asarray(disc_u, float)
    disc_v = np.asarray(disc_v, float)

    run = MonteCarloRun(
        run_number=0,
        sample_name="unit_test_curve",
        concordant_uPb=conc_u,
        concordant_pbPb=conc_v,
        discordant_uPb=disc_u,
        discordant_pbPb=disc_v,
        discordant_labels=None,
        settings=settings,
    )

    # Evaluate a small grid around the true Pb-loss age
    grid = np.array([1000e6, 1200e6, 1500e6, 1800e6, 2200e6], float)
    for t in grid:
        run.samplePbLossAge(float(t), test, penalise_invalid_ages=True)

    scores = np.array([run.statistics_by_pb_loss_age[float(t)].score for t in grid], float)
    assert np.all(np.isfinite(scores))
    assert np.ptp(scores) > 1e-6  # NOT flat

    min_score = float(np.min(scores))
    score_true = float(run.statistics_by_pb_loss_age[float(t_lo_true)].score)
    assert abs(score_true - min_score) < 1e-12

    run.calculateOptimalAge()
    assert math.isfinite(run.optimal_pb_loss_age)
    # If there’s a plateau, allow any age on the plateau; otherwise it should hit t_lo_true.
    plateau = [float(t) for t, s in zip(grid, scores) if abs(float(s) - min_score) < 1e-12]
    assert float(run.optimal_pb_loss_age) in plateau
