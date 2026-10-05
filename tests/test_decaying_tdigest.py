import math

import pytest

from tdigest import TDigest, ExponentialDecayingTDigest


@pytest.fixture()
def digest():
    return ExponentialDecayingTDigest(half_life=10.0)


def test_rejects_non_positive_half_life():
    with pytest.raises(ValueError):
        ExponentialDecayingTDigest(half_life=0)
    with pytest.raises(ValueError):
        ExponentialDecayingTDigest(half_life=-1)


def test_empty_digest_queries_return_none(digest):
    assert digest.is_empty
    assert digest.effective_count == 0
    assert digest.percentile(50) is None
    assert digest.cdf(1.0) is None
    assert digest.trimmed_mean(1, 99) is None


def test_matches_reference_with_explicit_decayed_weights():
    half_life = 10.0
    samples = [(0.0, float(v)) for v in range(0, 10)]
    samples += [(10.0, float(v)) for v in range(100, 110)]
    samples += [(25.0, float(v)) for v in range(200, 210)]
    t_ref = 25.0

    decaying = ExponentialDecayingTDigest(half_life=half_life)
    for t, x in samples:
        decaying.update(x, t)

    reference = TDigest()
    for t, x in samples:
        reference.update(x, 0.5 ** ((t_ref - t) / half_life))

    for p in (1, 10, 25, 50, 75, 90, 99):
        assert decaying.percentile(p, t=t_ref) == pytest.approx(
            reference.percentile(p), rel=1e-9, abs=1e-9)
    for x in (5.0, 105.0, 150.0, 205.0):
        assert decaying.cdf(x, t=t_ref) == pytest.approx(
            reference.cdf(x), rel=1e-9, abs=1e-9)


def test_recent_data_dominates_after_many_half_lives(digest):
    digest.batch_update([0.0] * 100, t=0.0)
    digest.batch_update([1000.0] * 100, t=100.0)  # 10 half-lives later
    # old samples retain only ~0.1% of the total weight
    assert digest.percentile(50) > 990.0
    assert digest.percentile(99) == pytest.approx(1000.0)
    assert digest.effective_count == pytest.approx(100 + 100 * 0.5 ** 10)


def test_time_must_not_go_backwards(digest):
    digest.update(1.0, t=5.0)
    with pytest.raises(ValueError):
        digest.update(2.0, t=4.0)
    with pytest.raises(ValueError):
        digest.percentile(50, t=4.999)
    # equal time is fine
    digest.update(2.0, t=5.0)
    assert digest.percentile(50, t=5.0) is not None


def test_query_after_long_idle_uses_same_scale(digest):
    digest.batch_update([float(v) for v in range(100)], t=0.0)
    p50_before = digest.percentile(50, t=10.0)
    p90_before = digest.percentile(90, t=10.0)
    count_before = digest.effective_count
    # long idle gap, no new data: decay must not change the distribution
    assert digest.percentile(50, t=60.0) == pytest.approx(p50_before)
    assert digest.percentile(90, t=60.0) == pytest.approx(p90_before)
    assert digest.effective_count == pytest.approx(count_before * 0.5 ** 5)


def test_underflow_of_decay_factor_gives_explicit_empty_state(digest):
    # half_life=10, jumping 20000 time units => 0.5**2000 underflows to 0.0
    digest.batch_update([1.0, 2.0, 3.0], t=0.0)
    assert digest.percentile(50, t=20000.0) is None
    assert digest.is_empty
    assert digest.effective_count == 0
    assert digest.cdf(2.0) is None

    # new data after the gap must not resurrect old samples
    digest.update(100.0, t=20000.0)
    digest.update(101.0, t=20000.0)
    assert digest.percentile(50) == pytest.approx(100.5)
    for c in digest._digest.C.values():
        assert math.isfinite(c.count) and c.count > 0
        assert math.isfinite(c.mean)


def test_effective_weight_floor_resets_digest():
    d = ExponentialDecayingTDigest(half_life=1.0, min_effective_weight=0.5)
    d.update(1.0, t=0.0)  # effective weight 1.0
    # after 2 half-lives effective weight is 0.25 < 0.5 -> empty
    assert d.percentile(50, t=2.0) is None
    assert d.is_empty
    # old data must not reappear once new data arrives
    d.update(50.0, t=2.0)
    assert d.effective_count == pytest.approx(1.0)
    assert d.percentile(50) == pytest.approx(50.0)


def test_renormalization_preserves_relative_weights(digest):
    digest.batch_update([float(v) for v in range(50)], t=0.0)
    p50_before = digest.percentile(50)
    # force the renormalization path on the next decay step
    digest.RENORMALIZE_THRESHOLD = 1.0
    expected = {c.mean: c.count * digest._scale * 0.5
                for c in digest._digest.C.values()}
    # one half-life passes; the query triggers decay + renormalization
    p50_after = digest.percentile(50, t=10.0)
    assert digest._scale == 1.0
    for c in digest._digest.C.values():
        assert c.count == pytest.approx(expected[c.mean])
        assert c.count > 0 and math.isfinite(c.count)
    # distribution is unchanged by rescaling
    assert p50_after == pytest.approx(p50_before)


def test_extreme_time_jump_leaves_no_nan_or_negative_weights(digest):
    digest.batch_update([float(v) for v in range(1000)], t=0.0)
    digest.update(5.0, t=1e15)  # absurdly far future: full decay + reset
    assert not digest.is_empty  # the fresh sample at t=1e15 remains
    assert digest.percentile(50) == pytest.approx(5.0)
    for c in digest._digest.C.values():
        assert math.isfinite(c.count) and c.count > 0
        assert not math.isnan(c.mean)


def test_rejects_non_positive_weight(digest):
    with pytest.raises(ValueError):
        digest.update(1.0, t=0.0, w=0)
    with pytest.raises(ValueError):
        digest.update(1.0, t=0.0, w=-2)
