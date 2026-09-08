import numpy as np

from outlier_ai.outlier.stats import bootstrap_ratio_lower_bound, wilson_lower_bound
from outlier_ai.synthetic.cards import SEED_STRATEGY_CARDS
from outlier_ai.synthetic.comments import generate_comments, summarize_signals
from outlier_ai.synthetic.latent import AccountProfile, LatentOutcomeModel

CATS = ["skincare", "coffee", "apparel"]
SLUGS = [c.slug for c in SEED_STRATEGY_CARDS]


def test_latent_is_deterministic_per_seed():
    a = LatentOutcomeModel(1, CATS, SLUGS)
    b = LatentOutcomeModel(1, CATS, SLUGS)
    c = LatentOutcomeModel(2, CATS, SLUGS)
    assert a.affinity == b.affinity
    assert a.affinity != c.affinity
    assert len(a.hot_pairs()) == 6


def test_hot_pairs_have_higher_mean_than_default():
    m = LatentOutcomeModel(3, CATS, SLUGS)
    niche, cat = m.hot_pairs()[0]
    assert m.mean_multiple(niche, cat) > 3.0
    cold = next(k for k in [(n, c) for n in SLUGS for c in CATS] if k not in m.affinity)
    assert m.mean_multiple(*cold) < 1.2


def test_multiples_are_heavy_tailed_but_mostly_below_median():
    m = LatentOutcomeModel(4, CATS, SLUGS)
    rng = np.random.default_rng(0)
    cold = next(k for k in [(n, c) for n in SLUGS for c in CATS] if k not in m.affinity)
    samples = np.array(
        [m.sample_truth(cold[0], cold[1], "none", rng).roas_multiple for _ in range(4000)]
    )
    assert 0.6 < np.median(samples) < 1.1
    tail = (samples >= 3.0).mean()
    assert 0.001 < tail < 0.08


def test_simulate_day_is_internally_consistent():
    m = LatentOutcomeModel(5, CATS, SLUGS)
    rng = np.random.default_rng(1)
    truth = m.sample_truth(SLUGS[0], CATS[0], "none", rng)
    row = m.simulate_day(truth, 20.0, AccountProfile(), 0, "screening", rng)
    assert 0 <= row.link_clicks <= row.impressions
    assert row.spend <= 20.0
    assert row.revenue >= 0 and (row.revenue == 0) == (row.purchases == 0)
    scale_row = m.simulate_day(truth, 100.0, AccountProfile(), 6, "scale", rng)
    assert scale_row.frequency > row.frequency


def test_wilson_lower_bound_behaves():
    assert wilson_lower_bound(0, 0) == 0.0
    lb_small = wilson_lower_bound(12, 1000)
    lb_big = wilson_lower_bound(120, 10000)
    assert lb_small < 0.012 and lb_big < 0.012
    assert lb_big > lb_small  # more data, tighter bound
    assert wilson_lower_bound(1000, 1000) > 0.99


def test_bootstrap_ratio_lower_bound_is_below_point_estimate():
    rng = np.random.default_rng(0)
    rev = rng.uniform(150, 350, size=7)
    spd = np.full(7, 100.0)
    point = rev.sum() / spd.sum() / 2.0
    lb = bootstrap_ratio_lower_bound(rev, spd, baseline=2.0, n_samples=500, rng=rng)
    assert 0 < lb <= point
    assert bootstrap_ratio_lower_bound(np.array([300.0]), np.array([100.0]), 2.0) == 1.5


def test_comments_lean_negative_when_roas_is_low():
    rng = np.random.default_rng(0)
    low = generate_comments(rng, 400, roas_multiple=0.5, polarity=0.6, window_hours=168)
    high = generate_comments(rng, 400, roas_multiple=4.0, polarity=0.3, window_hours=168)
    low_neg = sum(c.theme_key == "price_objection" for c in low)
    high_neg = sum(c.theme_key == "price_objection" for c in high)
    assert low_neg > high_neg
    pii = [c for c in low + high if c.raw_text != c.clean_text]
    assert pii, "some comments should carry planted PII for the stripper"
    drafts = summarize_signals(low)
    assert drafts and drafts[0].count >= drafts[-1].count
