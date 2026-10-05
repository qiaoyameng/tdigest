import pytest
from numpy import random
from numpy import percentile
from numpy import bitwise_and
from numpy import testing
from accumulation_tree import AccumulationTree
from tdigest.tdigest import TDigest, Centroid, DigestDistribution


def RBTree(data):
    t = AccumulationTree(lambda centroid: centroid.count)
    for k, v in data:
        t.insert(k, v)
    return t


@pytest.fixture()
def empty_tdigest():
    return TDigest()


@pytest.fixture()
def example_positive_centroids():
    return RBTree([
        (0.5, Centroid(0.5, 1)),
        (1.1, Centroid(1.1, 1)),
        (1.5, Centroid(1.5, 1)),
    ])


@pytest.fixture()
def example_centroids():
    return RBTree([
        (-1.1, Centroid(-1.1, 1)),
        (-0.5, Centroid(-0.5, 1)),
        (0.1, Centroid(0.1, 1)),
        (1.5, Centroid(1.5, 1)),
    ])


@pytest.fixture()
def example_random_data():
    return random.randn(100)

@pytest.fixture()
def sample_dict():
    return {'K': 25, 'delta': 0.01, 'min': 0.0, 'max': 3.0, 'centroids': [{'c': 1.0, 'm': 0.0}, {'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}], 'n': 4}


@pytest.fixture()
def legacy_dict():
    # A summary serialized before exact extremes were tracked: no 'min'/'max'.
    return {'K': 25, 'delta': 0.01, 'centroids': [{'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}], 'n': 3}

@pytest.fixture()
def sample_centroid_list():
    return [{'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}]


class TestTDigest():

    def test_add_centroid(self, empty_tdigest, example_positive_centroids):
        empty_tdigest.C = example_positive_centroids
        new_centroid = Centroid(0.9, 1)
        empty_tdigest._add_centroid(new_centroid)
        assert (empty_tdigest.C - RBTree([
            (0.5, Centroid(0.5, 1)),
            (new_centroid.mean, new_centroid),
            (1.1, Centroid(1.1, 1)),
            (1.5, Centroid(1.5, 1)),
        ])).is_empty()

        last_centroid = Centroid(10., 1)
        empty_tdigest._add_centroid(last_centroid)
        assert (empty_tdigest.C - RBTree([
            (0.5, Centroid(0.5, 1)),
            (new_centroid.mean, new_centroid),
            (1.1, Centroid(1.1, 1)),
            (1.5, Centroid(1.5, 1)),
            (last_centroid.mean, last_centroid),
        ])).is_empty()

    def test_add_centroid_if_key_already_present(self, empty_tdigest, example_positive_centroids):
        empty_tdigest.C = example_positive_centroids
        new_centroid = Centroid(1.1, 5)
        empty_tdigest._add_centroid(new_centroid)
        assert (empty_tdigest.C - RBTree([
            (0.5, Centroid(0.5, 1)),
            (1.1, Centroid(1.1, 1 + 5)),
            (1.5, Centroid(1.5, 1)),
        ])).is_empty()

    def test_compute_centroid_quantile(self, empty_tdigest, example_centroids):
        empty_tdigest.C = example_centroids
        empty_tdigest.n = 4

        assert empty_tdigest._compute_centroid_quantile(example_centroids[-1.1]) == (1 / 2. + 0) / 4
        assert empty_tdigest._compute_centroid_quantile(example_centroids[-0.5]) == (1 / 2. + 1) / 4
        assert empty_tdigest._compute_centroid_quantile(example_centroids[0.1]) == (1 / 2. + 2) / 4
        assert empty_tdigest._compute_centroid_quantile(example_centroids[1.5]) == (1 / 2. + 3) / 4

    def test_find_closest_centroids_works_with_positive_values(self, empty_tdigest, example_positive_centroids):
        empty_tdigest.C = example_positive_centroids
        assert empty_tdigest._find_closest_centroids(0.0) == [example_positive_centroids[0.5]]
        assert empty_tdigest._find_closest_centroids(2.0) == [example_positive_centroids[1.5]]
        assert empty_tdigest._find_closest_centroids(1.1) == [example_positive_centroids[1.1]]
        assert empty_tdigest._find_closest_centroids(1.2) == [example_positive_centroids[1.1]]
        assert empty_tdigest._find_closest_centroids(1.4) == [example_positive_centroids[1.5]]
        assert empty_tdigest._find_closest_centroids(1.3) == [example_positive_centroids[1.5],
                                                              example_positive_centroids[1.1]]

    def test_get_closest_centroids_works_with_negative_values(self, empty_tdigest, example_centroids):
        empty_tdigest.C = example_centroids
        assert empty_tdigest._find_closest_centroids(0.0) == [example_centroids[0.1]]
        assert empty_tdigest._find_closest_centroids(-2.0) == [example_centroids[-1.1]]
        assert empty_tdigest._find_closest_centroids(-0.6) == [example_centroids[-0.5]]
        assert empty_tdigest._find_closest_centroids(-0.4) == [example_centroids[-0.5]]

    def test_compress(self, empty_tdigest, example_random_data):
        empty_tdigest.batch_update(example_random_data)
        precompress_n, precompress_len = empty_tdigest.n, len(empty_tdigest)
        empty_tdigest.compress()
        postcompress_n, postcompress_len = empty_tdigest.n, len(empty_tdigest)
        assert postcompress_n == precompress_n
        assert postcompress_len <= precompress_len

    def test_data_comes_in_sorted_does_not_blow_up(self, empty_tdigest):
        t = TDigest()
        for x in range(10000):
            t.update(x, 1)

        assert len(t) < 5000

        t = TDigest()
        t.batch_update(range(10000))
        assert len(t) < 1000

    def test_extreme_percentiles_return_min_and_max(self, empty_tdigest):
        t = TDigest()
        data = random.randn(10000)
        t.batch_update(data)
        assert t.percentile(100.) == data.max()
        assert t.percentile(0) == data.min()
        assert t.percentile(0.1) > data.min()
        assert t.percentile(0.999) < data.max()

    def test_negative_extreme_percentile_is_still_positive(self, empty_tdigest):
        # Test https://github.com/CamDavidsonPilon/tdigest/issues/16
        t = TDigest()
        t.batch_update([62.0, 202.0, 1415.0, 1433.0])
        assert t.percentile(26) > 0

    def test_percentile_at_border_returns_an_intermediate_value(self, empty_tdigest):
        data = [62.0, 202.0, 1415.0, 1433.0]
        t = TDigest()
        t.batch_update(data)
        assert t.percentile(25) == 132.0

    def test_percentile_weighted_interpolation(self, empty_tdigest):
        data = [1, 10, 10, 10, 10, 100, 100, 100, 100, 1000]
        t = TDigest()
        t.batch_update(data)
        assert t.percentile(40) == 32.5

    def test_adding_centroid_with_existing_key_does_not_break_synchronicity(self, empty_tdigest, example_centroids):
        td = empty_tdigest
        td.C = example_centroids
        assert -1.1 in td.C
        td._add_centroid(Centroid(-1.1, 10))
        assert all([k == centroid.mean for k, centroid in td.C.items()])

    def test_cdf_with_single_centroid(self, empty_tdigest):
        td = empty_tdigest
        td.update(1)
        assert td.cdf(1) == 1

    def test_cdf_with_single_centroid_at_zero(self, empty_tdigest):
        td = empty_tdigest
        td.update(0)
        assert td.cdf(0) == 1

    def test_to_dict(self, empty_tdigest, sample_dict):
        td = empty_tdigest
        td.update(0)
        td.update(1)
        td.update(2)
        td.update(3)
        assert td.to_dict() == sample_dict

    def test_from_dict(self, empty_tdigest, sample_dict):
        td0 = TDigest().update_from_dict(sample_dict)
        td1 = TDigest.from_dict(sample_dict)
        for val in [-2,4,17]:
            assert abs(td0.cdf(val)-td1.cdf(val))<0.0001

    def test_from_dict_random(self, example_random_data):
        import numpy as np
        t = TDigest()
        t.batch_update(example_random_data)
        td = t.to_dict()
        td0 = TDigest().update_from_dict(td)
        td1 = TDigest.from_dict(td)
        for val in [0.2,0.5,1.4]:
            assert abs(td0.cdf(val)-td1.cdf(val))<0.0001

    def test_centroids_to_list(self, empty_tdigest, sample_centroid_list):
        td = empty_tdigest
        td.update(1)
        td.update(2)
        td.update(3)
        assert td.centroids_to_list() == sample_centroid_list

    def test_update_from_dict(self, empty_tdigest, sample_dict):
        td = empty_tdigest
        assert td.update_from_dict(sample_dict).to_dict() == sample_dict

    def test_update_centroids_from_list(self, empty_tdigest, sample_centroid_list):
        td = empty_tdigest
        assert td.update_centroids_from_list(sample_centroid_list).centroids_to_list() == sample_centroid_list

class TestDigestDistribution():

    def test_exact_extremes_survive_compression(self):
        t = TDigest()
        data = random.randn(10000)
        t.batch_update(data)
        t.compress()
        dist = t.distribution()
        assert isinstance(dist, DigestDistribution)
        assert dist.min_value == data.min()
        assert dist.max_value == data.max()
        assert t.percentile(0) == data.min()
        assert t.percentile(100) == data.max()
        # F(min) is at most half the outermost centroid's weight: the lower
        # half of that centroid's mass sits exactly at its mean
        assert t.cdf(data.min()) <= 1.0 / len(data)
        assert t.cdf(data.max()) == 1
        # no mass is placed outside the observed extremes
        assert t.cdf(data.min() - 1) == 0
        assert t.cdf(data.max() + 1) == 1

    def test_weighted_known_data(self):
        t = TDigest()
        t.update(1, 1)
        t.update(2, 3)
        t.update(3, 1)
        dist = t.distribution()
        assert dist.n == 5
        assert t.percentile(0) == 1
        assert t.percentile(100) == 3
        assert t.percentile(50) == 2
        assert t.cdf(2) == 0.5
        # the known minimum coincides with the first centroid mean, so the
        # cdf jumps there from 0 to half the first centroid's weight
        assert t.cdf(1) == 0.1
        assert t.cdf(3) == 1

    def test_repeated_values_single_centroid(self):
        t = TDigest()
        t.batch_update([5, 5, 5, 5])
        assert len(t) == 1
        for p in [0, 25, 50, 75, 100]:
            assert t.percentile(p) == 5
        assert t.cdf(5) == 1
        assert t.cdf(4.999) == 0

    def test_single_point_summary(self):
        t = TDigest()
        t.update(7)
        assert t.percentile(0) == 7
        assert t.percentile(50) == 7
        assert t.percentile(100) == 7
        assert t.cdf(7) == 1
        assert t.cdf(6) == 0

    def test_jump_uses_lower_generalized_inverse(self):
        # repeated values collapse into one centroid, creating a jump in the
        # cdf at 1 from 0 to 3/8 of the total weight
        t = TDigest()
        t.batch_update([1, 1, 1, 2])
        # F is right-continuous: the value at the jump is the top of the jump
        assert t.cdf(1) == 0.375
        # quantile is the lower generalized inverse Q(p) = inf{x: F(x) >= p},
        # so every p inside the jump maps back to the jump location
        assert t.percentile(0) == 1
        assert t.percentile(10) == 1
        assert t.percentile(37.5) == 1
        # the two maps are compatible at the jump
        assert t.percentile(t.cdf(1) * 100) == 1
        assert t.cdf(t.percentile(20)) >= 0.2

    def test_monotonicity_and_inverse_relation(self):
        t = TDigest()
        t.batch_update([1, 2, 2, 3, 3, 3, 4, 10])

        ps = [i / 100. for i in range(101)]
        quantiles = [t.percentile(100 * p) for p in ps]
        assert all(a <= b for a, b in zip(quantiles, quantiles[1:]))

        xs = [1 + 0.5 * i for i in range(19)]  # within [min, max]
        cdfs = [t.cdf(x) for x in xs]
        assert all(a <= b for a, b in zip(cdfs, cdfs[1:]))

        for p in ps:
            assert t.cdf(t.percentile(100 * p)) >= p - 1e-9
        for x in xs:
            assert t.percentile(100 * t.cdf(x)) <= x + 1e-9

    def test_merge_preserves_extremes(self):
        t1 = TDigest()
        t1.batch_update(random.randn(5000))
        t2 = TDigest()
        t2.batch_update(random.randn(5000) + 100)
        merged = t1 + t2
        assert merged.percentile(0) == min(t1.percentile(0), t2.percentile(0))
        assert merged.percentile(100) == max(t1.percentile(100), t2.percentile(100))

    def test_legacy_import_marks_extremes_unknown(self, legacy_dict):
        t = TDigest.from_dict(legacy_dict)
        dist = t.distribution()
        # extremes are unknown, not fabricated from the outermost centroids
        assert dist.min_value is None
        assert dist.max_value is None
        # quantiles at 0/100 fall back to the outermost centroid means as
        # estimates, and cdf/quantile stay mutually consistent
        assert t.percentile(0) == 1.0
        assert t.percentile(100) == 3.0
        assert t.cdf(2) == 0.5
        assert t.percentile(t.cdf(2) * 100) == 2.0

    def test_update_from_legacy_dict_marks_extremes_unknown(self, legacy_dict):
        t = TDigest().update_from_dict(legacy_dict)
        assert t.distribution().min_value is None
        assert t.distribution().max_value is None

    def test_merge_with_unknown_extremes_stays_unknown(self, legacy_dict):
        t_legacy = TDigest.from_dict(legacy_dict)
        t_known = TDigest()
        t_known.batch_update([1, 2, 3])
        merged = t_known + t_legacy
        assert merged.distribution().min_value is None
        assert merged.distribution().max_value is None

    def test_extremes_round_trip_through_dict(self):
        t = TDigest()
        t.batch_update(random.randn(1000))
        restored = TDigest.from_dict(t.to_dict())
        assert restored.percentile(0) == t.percentile(0)
        assert restored.percentile(100) == t.percentile(100)


class TestStatisticalTests():

    def test_uniform(self):
        t = TDigest()
        x = random.random(size=10000)
        t.batch_update(x)

        assert abs(t.percentile(50) - 0.5) < 0.01
        assert abs(t.percentile(10) - .1) < 0.01
        assert abs(t.percentile(90) - 0.9) < 0.01
        assert abs(t.percentile(1) - 0.01) < 0.005
        assert abs(t.percentile(99) - 0.99) < 0.005
        assert abs(t.percentile(0.1) - 0.001) < 0.001
        assert abs(t.percentile(99.9) - 0.999) < 0.001

    def test_ints(self):
        t = TDigest()
        t.batch_update([1, 2, 3])
        assert abs(t.percentile(50) - 2) < 0.0001

        t = TDigest()
        x = [1, 2, 2, 2, 2, 2, 2, 2, 3]
        t.batch_update(x)
        assert t.percentile(50) == 2
        assert sum([c.count for c in t.C.values()]) == len(x)

        t = TDigest()
        t.batch_update([1, 1, 2, 2, 3, 4, 4, 4, 5, 5])
        assert t.percentile(30) == 2
        assert t.percentile(40)*3 == 8

    @pytest.mark.parametrize("percentile_range", [[0, 7], [27, 47], [39, 66], [81, 99], [77, 100], [0, 100]])
    @pytest.mark.parametrize("data_size", [100, 1000, 5000])
    def test_trimmed_mean(self, percentile_range, data_size):
        p1 = percentile_range[0]
        p2 = percentile_range[1]

        t = TDigest()
        x = random.random(size=data_size)
        t.batch_update(x)

        tm_actual = t.trimmed_mean(p1, p2)
        tm_expected = x[bitwise_and(x >= percentile(x, p1), x <= percentile(x, p2))].mean()

        testing.assert_allclose(tm_actual, tm_expected, rtol=0.01, atol=0.01)

    def test_trimmed_mean_corner_cases(self):
        td = TDigest()

        mean = td.trimmed_mean(0, 100)
        assert mean == 0

        td.update(1)
        mean = td.trimmed_mean(0, 100)
        assert mean == 1

        td.update(1000)
        mean = td.trimmed_mean(0, 100)
        assert mean == 500.5

    def test_trimmed_mean_negative(self):
        td = TDigest()
        for i in range(100):
            td.update(random.random())

        for i in range(10):
            td.update(i*100)

        mean = td.trimmed_mean(1, 99)
        assert mean >= 0

class TestCentroid():

    def test_update(self):
        c = Centroid(0, 0)
        value, weight = 1, 1
        c.update(value, weight)
        assert c.count == 1
        assert c.mean == 1

        value, weight = 2, 1
        c.update(value, weight)
        assert c.count == 2
        assert c.mean == (2 + 1.) / 2.

        value, weight = 1, 2
        c.update(value, weight)
        assert c.count == 4
        assert c.mean == 1 * 1 / 4. + 2 * 1 / 4. + 1 * 2 / 4.
