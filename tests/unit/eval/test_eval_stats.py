from __future__ import annotations

import numpy as np

from outlier_ai.eval.stats import bootstrap_rate, intervals_disjoint, paired_difference


def test_bootstrap_rate_brackets_the_mean():
    ci = bootstrap_rate([1, 0, 0, 1, 1, 0, 0, 0, 1, 0], seed=1)
    assert ci.n == 10 and abs(ci.rate - 0.4) < 1e-9
    assert ci.low <= ci.rate <= ci.high
    assert 0.0 <= ci.low < ci.high <= 1.0
    empty = bootstrap_rate([], seed=1)
    assert empty.n == 0


def test_paired_difference_and_disjoint():
    a = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1] * 3)
    b = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1] * 3)
    d = paired_difference(a.tolist(), b.tolist(), seed=0)
    assert d.rate > 0.4 and d.low > 0
    hi = bootstrap_rate(a.tolist(), seed=0)
    lo = bootstrap_rate(b.tolist(), seed=0)
    assert intervals_disjoint(hi, lo)
    assert not intervals_disjoint(hi, hi)
