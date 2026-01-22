import math
import numpy as np

from process import calculations
from model.monteCarloRun import MonteCarloRun
from model.settings.calculation import ConcordiaMode


class _DummyDissimilarityTest:
    """Minimal stub matching your interface: perform() + getComparisonValue()."""
    def perform(self, a, b):
        # Return any valid KS-like pair (D, p)
        return 0.5, 1.0

    def getComparisonValue(self, test_statistics):
        # Your code uses this as "base" dissimilarity; mimic using D
        return float(test_statistics[0])


class _Settings:
    def __init__(self):
        self.concordiaMode = ConcordiaMode.WETHERILL


def test_discordant_age_wetherill_hits_upper_intercept():
    # Pick a lower intercept (Pb-loss) and an upper intercept
    t_lo = 1500e6
    t_hi = 3000e6

    # Wetherill concordia points
    x_lo = calculations.pb207u235_from_age(t_lo)
    y_lo = calculations.pb206u238_from_age(t_lo)
    x_hi = calculations.pb207u235_from_age(t_hi)
    y_hi = calculations.pb206u238_from_age(t_hi)

    # Point on the chord between the two concordia points
    f = 0.55
    x2 = x_hi + (x_lo - x_hi) * f
    y2 = y_hi + (y_lo - y_hi) * f

    ui = calculations.discordant_age_wetherill(x_lo, y_lo, x2, y2)
    assert ui is not None and math.isfinite(ui)

    # Should recover the upper intercept (within a small tolerance)
    assert abs(ui - t_hi) < 1e5  # 0.1 Ma


def test_montecarlo_run_wetherill_path_produces_finite_reconstructed_ages():
    settings = _Settings()
    test = _DummyDissimilarityTest()

    # Build some concordant points (TW coords derived from concordia ages)
    conc_ages = np.array([1600e6, 1700e6, 1800e6], float)
    conc_u = np.array([calculations.u238pb206_from_age(t) for t in conc_ages], float)
    conc_v = np.array([calculations.pb207pb206_from_age(t) for t in conc_ages], float)

    # Build discordant points as mixtures between Pb-loss age and an upper intercept
    t_lo = 1500e6
    t_hi = 3200e6
    u_lo = calculations.u238pb206_from_age(t_lo)
    v_lo = calculations.pb207pb206_from_age(t_lo)
    u_hi = calculations.u238pb206_from_age(t_hi)
    v_hi = calculations.pb207pb206_from_age(t_hi)

    fracs = np.array([0.2, 0.4, 0.6, 0.8], float)
    disc_u = u_hi + (u_lo - u_hi) * fracs
    disc_v = v_hi + (v_lo - v_hi) * fracs

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
    assert min(st.valid_discordant_ages) > t_lo  # upper intercept should be older than Pb-loss age
