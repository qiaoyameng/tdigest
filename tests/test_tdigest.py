import pytest
from numpy import random
from numpy import percentile
from numpy import bitwise_and
from numpy import testing
from accumulation_tree import AccumulationTree
from decimal import Decimal, getcontext

from tdigest import wasserstein_distance
from tdigest.tdigest import TDigest, Centroid


def digest_from_centroids(centroids, total_weight=None, delta=0.01, K=25):
    data = {
        'K': K,
        'delta': delta,
        'n': total_weight if total_weight is not None else sum(weight for _, weight in centroids),
        'centroids': [{'m': mean, 'c': weight} for mean, weight in centroids],
    }
    return TDigest.from_dict(data)


def reference_centroid_cdf(centroids, x, side='right'):
    means = [Decimal(str(mean)) for mean, _ in centroids]
    weights = [Decimal(str(weight)) for _, weight in centroids]
    x = Decimal(str(x))
    total_weight = sum(weights)
    anchors = []
    cumulative_weight = Decimal(0)

    for weight in weights:
        anchors.append((cumulative_weight + weight / 2) / total_weight)
        cumulative_weight += weight

    if x < means[0]:
        return Decimal(0)
    if x > means[-1]:
        return Decimal(1)

    for i, mean in enumerate(means):
        if x == mean:
            if len(means) == 1:
                return Decimal(1)
            if i == 0:
                return anchors[0] if side == 'right' else Decimal(0)
            if i == len(means) - 1:
                return Decimal(1) if side == 'right' else anchors[-1]
            return anchors[i]
        if x < mean:
            previous_mean = means[i - 1]
            previous_anchor = anchors[i - 1]
            width = mean - previous_mean
            return previous_anchor + (anchors[i] - previous_anchor) * (x - previous_mean) / width

    return Decimal(1)


def simpson_integrate(function, left, right):
    midpoint = (left + right) / 2
    return (right - left) * (
        function(left) + 4 * function(midpoint) + function(right)
    ) / 6


def adaptive_simpson(function, left, right, tolerance, depth=24):
    midpoint = (left + right) / 2
    whole = simpson_integrate(function, left, right)
    left_half = simpson_integrate(function, left, midpoint)
    right_half = simpson_integrate(function, midpoint, right)
    error = left_half + right_half - whole

    if depth == 0 or abs(error) <= 15 * tolerance:
        return left_half + right_half + error / 15

    return (
        adaptive_simpson(function, left, midpoint, tolerance / 2, depth - 1) +
        adaptive_simpson(function, midpoint, right, tolerance / 2, depth - 1)
    )


def reference_wasserstein_distance(first_centroids, second_centroids):
    getcontext().prec = 50
    breakpoints = sorted(set(
        [Decimal(str(mean)) for mean, _ in first_centroids] +
        [Decimal(str(mean)) for mean, _ in second_centroids]
    ))
    distance = Decimal(0)

    for left, right in zip(breakpoints, breakpoints[1:]):
        def integrand(
            x,
            first_centroids=first_centroids,
            second_centroids=second_centroids,
            left=left,
            right=right
        ):
            first_side = 'right' if x == left else 'left' if x == right else 'right'
            second_side = 'right' if x == left else 'left' if x == right else 'right'
            return abs(
                reference_centroid_cdf(first_centroids, x, first_side) -
                reference_centroid_cdf(second_centroids, x, second_side)
            )

        distance += adaptive_simpson(
            integrand, left, right, Decimal('1e-35')
        )

    return float(distance)


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
    return {'K': 25, 'delta': 0.01, 'centroids': [{'c': 1.0, 'm': 0.0}, {'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}], 'n': 4}

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

    def test_cdf_rejects_non_finite_value(self, empty_tdigest):
        td = empty_tdigest
        td.update(0)

        with pytest.raises(ValueError):
            td.cdf(float('nan'))

    def test_centroid_distribution_is_shared_interpolation_model(self):
        td = digest_from_centroids([(0, 2), (10, 2)])

        assert td.centroid_distribution() == [
            (0, 0, 0.25),
            (10, 0.75, 1),
        ]
        assert td.cdf(5) == 0.5
        assert td.percentile(50) == 5

    def test_wasserstein_distance_between_point_masses(self):
        first = digest_from_centroids([(0, 1)])
        second = digest_from_centroids([(5, 2)])

        assert first.wasserstein_distance(second) == 5
        assert second.wasserstein_distance(first) == 5
        assert wasserstein_distance(first, second) == 5
        assert wasserstein_distance(second, first) == 5

    def test_wasserstein_distance_normalizes_different_total_weights(self):
        centroids = [(0, 1), (2, 2), (5, 1), (9, 3)]
        scaled_centroids = [(mean, weight * 17) for mean, weight in centroids]

        first = digest_from_centroids(centroids)
        second = digest_from_centroids(scaled_centroids)

        assert first.wasserstein_distance(second) == 0

    def test_wasserstein_distance_is_order_independent(self):
        first = digest_from_centroids([(0, 1), (2, 2), (5, 1), (9, 3), (12, 1)])
        second = digest_from_centroids([(1, 2), (3, 1), (4, 4), (7, 2), (10, 2), (12, 2)])

        assert first.wasserstein_distance(second) == second.wasserstein_distance(first)

    def test_wasserstein_distance_of_shifted_summary_is_shift(self):
        centroids = [(0, 1), (1, 2), (3, 1), (6, 3), (10, 1)]
        shift = Decimal('1.5')
        shifted = [(float(Decimal(str(mean)) + shift), weight) for mean, weight in centroids]

        first = digest_from_centroids(centroids)
        second = digest_from_centroids(shifted)

        assert first.wasserstein_distance(second) == pytest.approx(1.5)
        assert first.wasserstein_distance(second) == pytest.approx(
            reference_wasserstein_distance(centroids, shifted), abs=1e-12
        )

    def test_wasserstein_distance_matches_numerical_integration(self):
        first_centroids = [(0, 1), (2, 2), (5, 1), (9, 3), (12, 1)]
        second_centroids = [(1, 2), (3, 1), (4, 4), (7, 2), (10, 2), (12, 2)]
        first = digest_from_centroids(first_centroids, delta=0.05, K=25)
        second = digest_from_centroids(second_centroids, delta=0.005, K=50)

        naive_equal_weight_difference = 4. / 3
        assert first.wasserstein_distance(second) == pytest.approx(
            reference_wasserstein_distance(first_centroids, second_centroids), abs=1e-12
        )
        assert first.wasserstein_distance(second) != pytest.approx(naive_equal_weight_difference)

    @pytest.mark.parametrize('centroids', [
        [],
        [(float('nan'), 1)],
        [(float('inf'), 1)],
        [(1, float('nan'))],
        [(1, float('-inf'))],
        [(1, 0)],
        [(1, -1)],
    ])
    def test_wasserstein_distance_rejects_invalid_summaries(self, centroids):
        total_weight = sum(weight for _, weight in centroids)
        first = digest_from_centroids([(0, 1)])
        second = digest_from_centroids(centroids, total_weight=total_weight)

        with pytest.raises(ValueError):
            wasserstein_distance(first, second)

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
