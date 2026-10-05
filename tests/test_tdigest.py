import random as py_random
import itertools
import pytest
from numpy import random
from numpy import percentile
from numpy import bitwise_and
from numpy import testing
from accumulation_tree import AccumulationTree
from tdigest.tdigest import TDigest, Centroid, merge_many


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


def _raw_digest(mean_weight_pairs, delta=0.01, K=25):
    """Build a TDigest directly from (mean, weight) pairs, no merging."""
    t = TDigest(delta, K)
    for mean, weight in mean_weight_pairs:
        t.C.insert(mean, Centroid(mean, weight))
        t.n += weight
    return t


class TestMergeMany():

    def _shards(self, seed=7, n_shards=5, size=1000):
        py_random.seed(seed)
        shards = []
        for _ in range(n_shards):
            t = TDigest()
            t.batch_update([py_random.gauss(0, 1) for _ in range(size)])
            shards.append(t)
        return shards

    def test_returns_tdigest(self):
        assert isinstance(merge_many(self._shards(n_shards=2, size=50)), TDigest)

    def test_permutation_invariant(self):
        shards = self._shards()
        reference = merge_many(shards).to_dict()
        for permutation in itertools.permutations(shards):
            assert merge_many(list(permutation)).to_dict() == reference

    def test_determinism_repeated_calls(self):
        shards = self._shards()
        first = merge_many(shards).to_dict()
        second = merge_many(list(reversed(shards))).to_dict()
        assert first == second

    def test_reproducible_after_save_and_reload(self):
        shards = self._shards()
        merged = merge_many(shards)

        reloaded_shards = [TDigest.from_dict(s.to_dict()) for s in shards]
        merged_reloaded = merge_many(reloaded_shards)

        assert merged_reloaded.to_dict() == merged.to_dict()

        once_more = TDigest.from_dict(merged.to_dict())
        assert once_more.to_dict() == merged.to_dict()

    def test_total_weight_preserved(self):
        shards = self._shards()
        expected_n = sum(s.n for s in shards)
        merged = merge_many(shards)
        assert merged.n == expected_n
        assert sum(c['c'] for c in merged.centroids_to_list()) == expected_n

    def test_extreme_values_preserved(self):
        shards = self._shards()
        all_centroids = [c for s in shards for c in s.centroids_to_list()]
        global_min = min(c['m'] for c in all_centroids)
        global_max = max(c['m'] for c in all_centroids)

        merged = merge_many(shards)
        assert merged.percentile(0) == global_min
        assert merged.percentile(100) == global_max

    def test_same_mean_weights_are_combined(self):
        a = _raw_digest([(1.0, 2.0), (3.0, 1.0)])
        b = _raw_digest([(1.0, 3.0), (5.0, 4.0)])
        merged = merge_many([a, b])

        by_mean = {c['m']: c['c'] for c in merged.centroids_to_list()}
        assert by_mean[1.0] == 5.0
        assert merged.n == 10.0
        assert sum(by_mean.values()) == 10.0

    def test_inputs_are_not_modified(self):
        shards = self._shards(n_shards=3, size=200)
        snapshots = [s.to_dict() for s in shards]
        merge_many(shards)
        assert [s.to_dict() for s in shards] == snapshots

    def test_does_not_mutate_global_random_state(self):
        shards = self._shards()
        py_random.seed(999)
        state = py_random.getstate()
        expected = py_random.random()
        py_random.setstate(state)

        merge_many(shards, random_state=1234)
        merge_many(shards, random_state=py_random.Random(7))

        assert py_random.random() == expected

    def test_random_state_does_not_change_result(self):
        shards = self._shards()
        reference = merge_many(shards).to_dict()
        assert merge_many(shards, random_state=1).to_dict() == reference
        assert merge_many(shards, random_state=2).to_dict() == reference
        assert merge_many(shards, random_state=py_random.Random(3)).to_dict() == reference

    def test_common_parameters_are_used(self):
        a = TDigest(delta=0.05, K=10); a.update(1)
        b = TDigest(delta=0.05, K=10); b.update(2)
        merged = merge_many([a, b])
        assert merged.delta == 0.05
        assert merged.K == 10

    def test_mixed_delta_falls_back_to_default(self):
        a = TDigest(delta=0.05, K=10); a.update(1)
        b = TDigest(delta=0.02, K=10); b.update(2)
        merged = merge_many([a, b])
        assert merged.delta == 0.01
        assert merged.K == 10

    def test_mixed_K_falls_back_to_default(self):
        a = TDigest(delta=0.05, K=10); a.update(1)
        b = TDigest(delta=0.05, K=20); b.update(2)
        merged = merge_many([a, b])
        assert merged.delta == 0.05
        assert merged.K == 25

    def test_explicit_parameters_override(self):
        a = TDigest(delta=0.05, K=10); a.update(1)
        b = TDigest(delta=0.02, K=20); b.update(2)
        merged = merge_many([a, b], delta=0.002, K=99)
        assert merged.delta == 0.002
        assert merged.K == 99

    def test_empty_iterable(self):
        merged = merge_many([])
        assert merged.n == 0
        assert merged.centroids_to_list() == []

    def test_empty_shards_are_ignored(self):
        a = TDigest()
        b = self._shards(n_shards=1, size=300)[0]
        assert merge_many([a, b]).to_dict() == merge_many([b]).to_dict()

    def test_negative_weight_rejected(self):
        bad = _raw_digest([(1.0, -2.0)])
        with pytest.raises(ValueError):
            merge_many([bad])

    def test_non_finite_mean_rejected(self):
        for bad_mean in (float('nan'), float('inf'), float('-inf')):
            bad = _raw_digest([(bad_mean, 1.0)])
            with pytest.raises(ValueError):
                merge_many([bad])

    def test_non_tdigest_input_rejected(self):
        with pytest.raises(TypeError):
            merge_many([1, 2])

    def test_invalid_explicit_parameters_rejected(self):
        a = TDigest(); a.update(1)
        with pytest.raises(ValueError):
            merge_many([a], delta=-1)
        with pytest.raises(ValueError):
            merge_many([a], K=0)

    def test_k_delta_cap_deterministic_and_weight_preserving(self):
        py_random.seed(3)
        shards = []
        for _ in range(4):
            t = TDigest(delta=0.5, K=5)
            t.batch_update([py_random.random() for _ in range(500)])
            shards.append(t)

        merged = merge_many(shards)
        assert len(merged) <= int(5 / 0.5)
        assert merged.n == 2000
        assert sum(c['c'] for c in merged.centroids_to_list()) == 2000

        reference = merged.to_dict()
        for permutation in itertools.islice(itertools.permutations(shards), 6):
            assert merge_many(list(permutation)).to_dict() == reference

    def test_percentile_accuracy_uniform(self):
        py_random.seed(0)
        shards = []
        for _ in range(4):
            t = TDigest()
            t.batch_update([py_random.random() for _ in range(2500)])
            shards.append(t)
        merged = merge_many(shards)
        assert abs(merged.percentile(50) - 0.5) < 0.01
        assert abs(merged.percentile(1) - 0.01) < 0.005
        assert abs(merged.percentile(99) - 0.99) < 0.005
