import math

import pytest

from tdigest.tdigest import (
    TDigest,
    ExponentiallyDecayingTDigest,
    EmptyDigestError,
)


HALF_LIFE = 10.0
LN2 = math.log(2.0)


def decay_weight(weight, insert_time, query_time, half_life):
    """Reference decay: weight * 0.5 ** (dt / half_life), computed explicitly."""
    return weight * math.exp(-LN2 * (query_time - insert_time) / half_life)


def reference_digest(samples, query_time, half_life, delta=0.01, K=25):
    """
    Build a plain TDigest whose centroid weights are the explicitly
    recomputed decayed weights at query_time. ``samples`` is a list of
    (value, insert_time, weight) tuples.
    """
    td = TDigest(delta=delta, K=K)
    for value, insert_time, weight in samples:
        decayed = decay_weight(weight, insert_time, query_time, half_life)
        if decayed > 0:
            td.update(value, decayed)
    return td


def make_groups():
    """
    Several groups of equal-time samples at well separated fixed times.
    Groups have distinct, widely separated value ranges so individual group
    weights can be inspected without centroid merging across groups.
    """
    groups = []
    base = 0
    for group_index, t in enumerate([0.0, 5.0, 10.0, 25.0]):
        for k in range(10):
            groups.append((base + k, t, 1.0))
        base += 10000
    return groups


def build_wrapper(groups, half_life=HALF_LIFE):
    d = ExponentiallyDecayingTDigest(half_life)
    for value, t, weight in groups:
        d.update(value, t, weight)
    return d


class TestExponentialDecayAgainstReference:

    @pytest.mark.parametrize("query_time", [0.0, 5.0, 10.0, 25.0, 30.0, 100.0])
    def test_effective_weight_matches_reference(self, query_time):
        groups = make_groups()
        visible = [s for s in groups if s[1] <= query_time]
        d = build_wrapper(visible)

        expected = sum(
            decay_weight(w, t, query_time, HALF_LIFE) for _, t, w in visible
        )
        assert d.effective_weight(query_time) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("query_time", [0.0, 5.0, 10.0, 25.0, 30.0, 100.0])
    def test_effective_centroid_weights_match_reference(self, query_time):
        # Groups have value ranges spaced 10000 apart, so no centroid can ever
        # merge across groups. Sum the centroid weights falling in each
        # group's range and compare against the explicit decay formula.
        groups = make_groups()
        visible = [s for s in groups if s[1] <= query_time]
        d = build_wrapper(visible)

        centroids = d.effective_centroids(query_time)
        for group_index, t in enumerate([0.0, 5.0, 10.0, 25.0]):
            if t > query_time:
                continue
            group_weight = sum(
                c for mean, c in centroids
                if group_index * 10000 <= mean < (group_index + 1) * 10000
            )
            expected = decay_weight(10.0, t, query_time, HALF_LIFE)
            assert group_weight == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("query_time", [0.0, 7.5, 25.0, 60.0, 100.0])
    @pytest.mark.parametrize("p", [1, 10, 25, 50, 75, 90, 99])
    def test_percentiles_match_reference(self, query_time, p):
        groups = make_groups()
        visible = [s for s in groups if s[1] <= query_time]
        d = build_wrapper(visible)
        ref = reference_digest(visible, query_time, HALF_LIFE)

        assert d.percentile(p, query_time) == pytest.approx(
            ref.percentile(p), rel=1e-9, abs=1e-6
        )

    @pytest.mark.parametrize("query_time", [0.0, 7.5, 25.0, 60.0, 100.0])
    def test_cdf_matches_reference(self, query_time):
        groups = make_groups()
        visible = [s for s in groups if s[1] <= query_time]
        d = build_wrapper(visible)
        ref = reference_digest(visible, query_time, HALF_LIFE)

        for x in [-5.0, 0.0, 4.5, 5000.0, 10000.0, 25000.0, 40000.0]:
            assert d.cdf(x, query_time) == pytest.approx(
                ref.cdf(x), abs=1e-9
            )

    def test_weight_halves_each_half_life(self):
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        d.update(1.0, 0.0, 1.0)
        assert d.effective_weight(0.0) == pytest.approx(1.0)
        assert d.effective_weight(HALF_LIFE) == pytest.approx(0.5)
        assert d.effective_weight(2 * HALF_LIFE) == pytest.approx(0.25)
        assert d.effective_weight(10 * HALF_LIFE) == pytest.approx(
            0.5 ** 10
        )


class TestLazyRenormalization:

    def test_no_centroid_traversal_needed_on_plain_updates(self):
        # Short, regular updates never trigger renormalization: results are
        # identical to the explicit-weight reference.
        groups = [(float(k), t, 1.0)
                  for t in range(10)
                  for k in range(20 * t, 20 * t + 20)]
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        for value, t, weight in groups:
            d.update(value, t, weight)

        for query_time in [9.0, 10.0, 20.0, 50.0]:
            ref = reference_digest(groups, query_time, HALF_LIFE)
            assert d.effective_weight(query_time) == pytest.approx(
                sum(decay_weight(w, t, query_time, HALF_LIFE)
                    for _, t, w in groups),
                rel=1e-12,
            )
            assert d.percentile(50, query_time) == pytest.approx(
                ref.percentile(50), rel=1e-6, abs=1e-6
            )

    def test_results_consistent_across_renormalization(self):
        # ~332 half-lives in one gap forces the lazy renormalization path.
        groups = make_groups()
        d = build_wrapper(groups)

        query_time = 4000.0
        ref = reference_digest(groups, query_time, HALF_LIFE)

        assert d.effective_weight(query_time) == pytest.approx(
            sum(decay_weight(w, t, query_time, HALF_LIFE)
                for _, t, w in groups),
            rel=1e-12,
        )
        for p in [1, 25, 50, 75, 99]:
            assert d.percentile(p, query_time) == pytest.approx(
                ref.percentile(p), rel=1e-9, abs=1e-6
            )

        # Adding fresh data after the giant gap keeps old survivors decaying
        # on the same scale; a later update stays consistent.
        for k in range(20):
            d.update(50000 + k, query_time, 1.0)
        extended = groups + [(float(50000 + k), query_time, 1.0)
                             for k in range(20)]

        later = query_time + 3.0
        ref2 = reference_digest(extended, later, HALF_LIFE)
        assert d.effective_weight(later) == pytest.approx(
            sum(decay_weight(w, t, later, HALF_LIFE)
                for _, t, w in extended),
            rel=1e-12,
        )
        assert d.percentile(50, later) == pytest.approx(
            ref2.percentile(50), rel=1e-6, abs=1e-6
        )

    def test_no_negative_or_nan_weights_after_renormalization(self):
        groups = make_groups()
        d = build_wrapper(groups)
        d.update(999999.0, 4000.0, 1.0)

        for _, centroid in d.effective_centroids(4000.0):
            assert math.isfinite(centroid)
            assert centroid >= 0
        assert math.isfinite(d.effective_weight(4000.0))


class TestUnderflowAndEmptyState:

    def test_explicit_empty_state_before_any_update(self):
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        assert d.is_empty()
        assert d.effective_weight() == 0.0
        assert d.effective_centroids() == []
        with pytest.raises(EmptyDigestError):
            d.percentile(50)
        with pytest.raises(EmptyDigestError):
            d.cdf(0.0)
        with pytest.raises(EmptyDigestError):
            d.trimmed_mean(0, 100)

    def test_query_far_past_underflow_reports_empty(self):
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        d.update(1.0, 0.0, 1.0)

        assert not d.is_empty(1000.0)
        far_future = 20000.0  # 0.5 ** 2000 underflows float64
        assert decay_weight(1.0, 0.0, far_future, HALF_LIFE) == 0.0
        assert d.is_empty(far_future)
        assert d.effective_weight(far_future) == 0.0
        assert d.effective_centroids(far_future) == []
        with pytest.raises(EmptyDigestError):
            d.percentile(50, far_future)

    def test_dead_data_does_not_revive_after_update(self):
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        d.update(123.0, 0.0, 1.0)

        far_future = 20000.0
        d.update(999.0, far_future, 1.0)

        assert d.is_empty() is False
        assert len(d) == 1
        only_mean = d.effective_centroids(far_future)[0][0]
        assert only_mean == pytest.approx(999.0)

        # The old sample is gone for good: it cannot reappear and percentiles
        # of the fresh data are exact.
        assert d.percentile(0, far_future) == 999.0
        assert d.percentile(100, far_future) == 999.0

    def test_repeated_huge_gaps_never_revive_old_data(self):
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        d.update(1.0, 0.0, 1.0)
        d.update(2.0, 20000.0, 1.0)
        d.update(3.0, 40000.0, 1.0)

        assert len(d) == 1
        assert d.percentile(50, 40000.0) == 3.0
        for _, weight in d.effective_centroids(40000.0):
            assert math.isfinite(weight) and weight > 0


class TestTimeSemantics:

    def test_backwards_update_rejected(self):
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        d.update(1.0, 10.0)
        with pytest.raises(ValueError):
            d.update(1.0, 9.9)
        # Equal time is allowed.
        d.update(2.0, 10.0)

    def test_backwards_query_rejected(self):
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        d.update(1.0, 10.0)
        with pytest.raises(ValueError):
            d.percentile(50, 5.0)
        with pytest.raises(ValueError):
            d.effective_weight(5.0)

    def test_queries_after_long_idle_use_same_scale(self):
        # No updates for a long stretch: percentile answers must be exactly
        # the same at insert time and at the idle time (scale invariance),
        # while total weight follows the decay scale.
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        d.batch_update([1.0, 2.0, 3.0, 4.0], 0.0)

        assert d.percentile(50, 0.0) == d.percentile(50, 3333.0)
        assert d.cdf(2.0, 0.0) == d.cdf(2.0, 3333.0)
        assert d.effective_weight(3333.0) == pytest.approx(
            4.0 * 0.5 ** (3333.0 / HALF_LIFE), rel=1e-12
        )

    def test_invalid_construction_arguments(self):
        with pytest.raises(ValueError):
            ExponentiallyDecayingTDigest(0)
        with pytest.raises(ValueError):
            ExponentiallyDecayingTDigest(-1.0)
        with pytest.raises(ValueError):
            ExponentiallyDecayingTDigest(float("nan"))

    def test_invalid_update_arguments(self):
        d = ExponentiallyDecayingTDigest(HALF_LIFE)
        with pytest.raises(ValueError):
            d.update(1.0, float("inf"))
        with pytest.raises(ValueError):
            d.update(1.0, 0.0, w=0)
        with pytest.raises(ValueError):
            d.update(1.0, 0.0, w=-1.0)
        with pytest.raises(ValueError):
            d.update(float("nan"), 0.0)
