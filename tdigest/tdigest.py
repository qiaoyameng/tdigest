from __future__ import print_function

import math
import random
from random import choice
from itertools import chain
from accumulation_tree import AccumulationTree
import pyudorandom


def _centroid_count(centroid):
    # this is defined at the top level because
    # pyspark needs to pickle these functions properly
    # See https://github.com/CamDavidsonPilon/tdigest/issues/42
    return centroid.count


# Default compression parameters used by merge_many when the input
# digests do not agree on a single set. See merge_many for the convention.
DEFAULT_MERGE_DELTA = 0.01
DEFAULT_MERGE_K = 25


def _validate_merge_params(delta, K):
    if not (isinstance(delta, (int, float)) and math.isfinite(delta)) or delta <= 0:
        raise ValueError("delta must be a finite, positive number.")
    if not (isinstance(K, (int, float)) and math.isfinite(K)) or K <= 0:
        raise ValueError("K must be a finite, positive number.")


def _merge_many_centroids(weighted_centroids, delta, K):
    """
    Deterministic, order-independent clustering of already-collected
    ``(mean, count)`` pairs.

    The input pairs are normalised first: equal means are combined by adding
    their weights and the result is sorted by mean. Compression then walks the
    centroids strictly left-to-right, so the same multiset of centroids always
    produces the same digest regardless of the order it arrived in.
    """
    combined = {}
    total_weight = 0.0
    for mean, count in weighted_centroids:
        combined[mean] = combined.get(mean, 0.0) + count
        total_weight += count

    result = TDigest(delta, K)
    result.n = total_weight
    if not combined:
        return result

    sorted_items = sorted(combined.items())

    # Canonical deterministic t-digest merge (Dunning & Ertl). Every
    # centroid's allowed size is derived from its own midpoint quantile using
    # the fixed scale function, and centroids are visited in mean order. This
    # is a pure function of the multiset of (mean, weight) pairs, so it is
    # independent of both shard order and in-shard traversal order.
    def k_limit(q):
        # Scale function k(q) = delta * Z(q); here Z(q) = 4 * n * q(1 - q),
        # matching the streaming update's _threshold convention.
        return 4.0 * total_weight * delta * q * (1.0 - q)

    def weight_limit(cumulative_before, cluster_count):
        midpoint_q = (cumulative_before + cluster_count / 2.0) / total_weight
        return k_limit(midpoint_q)

    cluster_means = []
    cluster_weights = []
    cluster_mean = 0.0
    cluster_count = 0.0
    cumulative = 0.0

    for mean, weight in sorted_items:
        proposed = cluster_count + weight
        limit = weight_limit(cumulative, proposed)
        if cluster_count > 0 and proposed > limit:
            cluster_means.append(cluster_mean)
            cluster_weights.append(cluster_count)
            cumulative += cluster_count
            cluster_mean = mean
            cluster_count = weight
        else:
            if cluster_count == 0:
                cluster_mean = mean
            else:
                cluster_mean = (
                    cluster_mean * cluster_count + mean * weight
                ) / proposed
            cluster_count = proposed
    if cluster_count > 0:
        cluster_means.append(cluster_mean)
        cluster_weights.append(cluster_count)

    # Enforce the K / delta cap as a hard upper bound with a deterministic
    # overflow merge. A centroid that legitimately exceeds its size limit is
    # retained rather than dropping weight.
    max_centroids = int(K / delta)
    if max_centroids > 0 and len(cluster_weights) > max_centroids:
        while len(cluster_weights) > max_centroids:
            # Merge the pair whose combined weight is smallest; tie-break on
            # the leftmost index so the choice is deterministic.
            best_index = 0
            best_cost = None
            for index in range(len(cluster_weights) - 1):
                cost = cluster_weights[index] + cluster_weights[index + 1]
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_index = index
            left_weight = cluster_weights[best_index]
            right_weight = cluster_weights[best_index + 1]
            merged_weight = left_weight + right_weight
            merged_mean = (
                cluster_means[best_index] * left_weight
                + cluster_means[best_index + 1] * right_weight
            ) / merged_weight
            cluster_means[best_index] = merged_mean
            cluster_weights[best_index] = merged_weight
            del cluster_means[best_index + 1]
            del cluster_weights[best_index + 1]

    for mean, weight in zip(cluster_means, cluster_weights):
        result.C.insert(mean, Centroid(mean, weight))
    return result


def merge_many(digests, delta=None, K=None, random_state=None):
    """
    Merge an iterable of :class:`TDigest` shards deterministically.

    Unlike repeatedly applying ``digest_a + digest_b`` -- which walks centroids
    in a shuffled order and makes random tie-breaking choices while mutating the
    global ``random`` module state -- this function gathers every centroid
    first and applies a fixed merging convention, so:

    - the same collection of shards yields the same result in any order;
    - the result is reproducible after ``to_dict`` / ``from_dict`` round trips;
    - the global ``random`` state is never read or modified.

    Parameters
    ----------
    digests : iterable of TDigest
        The shards to merge. Each shard is left untouched. Empty iterables and
        empty shards are allowed.
    delta, K : float, optional
        Output compression parameters. The convention for choosing them when
        the inputs disagree is explicit:

        * If ``delta`` / ``K`` are passed explicitly, those values are used.
        * Otherwise, if every non-empty input shares the same ``delta`` /
          ``K``, that common value is used.
        * Otherwise (mixed parameters, or no usable inputs) the library
          defaults (``delta=0.01``, ``K=25``) are used.

        The choice is made independently for ``delta`` and ``K``.
    random_state : int or random.Random, optional
        A local random source reserved for tie-breaking. Construction is fully
        deterministic and never reads or modifies the global ``random`` state;
        this argument lets callers pass an explicit, isolated source.

    Raises
    ------
    ValueError
        If any centroid has a negative weight or a non-finite (NaN/infinite)
        mean, or if the resulting ``delta`` / ``K`` are not finite/positive.
    TypeError
        If an input is not a :class:`TDigest`.

    Returns
    -------
    TDigest
        A new digest; the inputs are not modified.
    """
    digest_list = list(digests)
    for digest in digest_list:
        if not isinstance(digest, TDigest):
            raise TypeError("all inputs to merge_many must be TDigest instances.")

    # The merge is fully deterministic and does not draw randomness, but a
    # local source is still constructed for the documented contract. Building
    # it this way neither reads nor seeds the global ``random`` module state.
    rng = random_state if isinstance(random_state, random.Random) else random.Random(random_state)
    rng.seed(0)
    del rng

    non_empty = [d for d in digest_list if d.n > 0 or len(d.C) > 0]

    chosen_delta = delta
    if chosen_delta is None:
        deltas = {d.delta for d in non_empty}
        chosen_delta = next(iter(deltas)) if len(deltas) == 1 else DEFAULT_MERGE_DELTA

    chosen_K = K
    if chosen_K is None:
        Ks = {d.K for d in non_empty}
        chosen_K = next(iter(Ks)) if len(Ks) == 1 else DEFAULT_MERGE_K

    _validate_merge_params(chosen_delta, chosen_K)

    weighted_centroids = []
    for digest in digest_list:
        for centroid in list(digest.C.values()):
            mean = float(centroid.mean)
            count = float(centroid.count)
            if not math.isfinite(count) or count < 0:
                raise ValueError("centroid weights must be non-negative.")
            if not math.isfinite(mean):
                raise ValueError("centroid means must be finite.")
            if count > 0:
                weighted_centroids.append((mean, count))

    return _merge_many_centroids(weighted_centroids, chosen_delta, chosen_K)


class Centroid(object):

    def __init__(self, mean, count):
        self.mean = float(mean)
        self.count = float(count)

    def __repr__(self):
        return """<Centroid: mean=%.8f, count=%d>""" % (self.mean, self.count)

    def __eq__(self, other):
        return self.mean == other.mean and self.count == other.count

    def update(self, x, weight):
        self.count += weight
        self.mean += weight * (x - self.mean) / self.count
        return


class TDigest(object):

    def __init__(self, delta=0.01, K=25):

        self.C = AccumulationTree(_centroid_count)
        self.n = 0
        self.delta = delta
        self.K = K

    def __add__(self, other_digest):
        data = list(chain(self.C.values(), other_digest.C.values()))
        new_digest = TDigest(self.delta, self.K)

        if len(data) > 0:
            for c in pyudorandom.items(data):
                new_digest.update(c.mean, c.count)

        return new_digest

    def __len__(self):
        return len(self.C)

    def __repr__(self):
        return """<T-Digest: n=%d, centroids=%d>""" % (self.n, len(self))

    def __iter__(self):
        """
        Iterates over centroids in the digest.
        """
        return iter(self.centroids_to_list())

    def _add_centroid(self, centroid):
        if centroid.mean not in self.C:
            self.C.insert(centroid.mean, centroid)
        else:
            self.C[centroid.mean].update(centroid.mean, centroid.count)

    def _compute_centroid_quantile(self, centroid):
        denom = self.n
        cumulative_sum = self.C.get_left_accumulation(centroid.mean)
        return (centroid.count / 2. + cumulative_sum) / denom

    def _update_centroid(self, centroid, x, w):
        self.C.pop(centroid.mean)
        centroid.update(x, w)
        self._add_centroid(centroid)

    def _find_closest_centroids(self, x):
        try:
            ceil_key = self.C.ceiling_key(x)
        except KeyError:
            floor_key = self.C.floor_key(x)
            return [self.C[floor_key]]

        try:
            floor_key = self.C.floor_key(x)
        except KeyError:
            ceil_key = self.C.ceiling_key(x)
            return [self.C[ceil_key]]

        if abs(floor_key - x) < abs(ceil_key - x):
            return [self.C[floor_key]]
        elif abs(floor_key - x) == abs(ceil_key - x) and (ceil_key != floor_key):
            return [self.C[ceil_key], self.C[floor_key]]
        else:
            return [self.C[ceil_key]]

    def _threshold(self, q):
        return 4 * self.n * self.delta * q * (1 - q)

    def update(self, x, w=1):
        """
        Update the t-digest with value x and weight w.

        """
        self.n += w

        if len(self) == 0:
            self._add_centroid(Centroid(x, w))
            return

        S = self._find_closest_centroids(x)

        while len(S) != 0 and w > 0:
            j = choice(list(range(len(S))))
            c_j = S[j]

            q = self._compute_centroid_quantile(c_j)

            # This filters the out centroids that do not satisfy the second part
            # of the definition of S. See original paper by Dunning.
            if c_j.count + w > self._threshold(q):
                S.pop(j)
                continue

            delta_w = min(self._threshold(q) - c_j.count, w)
            self._update_centroid(c_j, x, delta_w)
            w -= delta_w
            S.pop(j)

        if w > 0:
            self._add_centroid(Centroid(x, w))

        if len(self) > self.K / self.delta:
            self.compress()

        return

    def batch_update(self, values, w=1):
        """
        Update the t-digest with an iterable of values. This assumes all points have the
        same weight.
        """
        for x in values:
            self.update(x, w)
        self.compress()
        return

    def compress(self):
        T = TDigest(self.delta, self.K)
        C = list(self.C.values())
        for c_i in pyudorandom.items(C):
            T.update(c_i.mean, c_i.count)
        self.C = T.C

    def percentile(self, p):
        """
        Computes the percentile of a specific value in [0,100].
        """

        if not (0 <= p <= 100):
            raise ValueError("p must be between 0 and 100, inclusive.")

        p = float(p)/100.
        p *= self.n
        c_i = None
        t = 0

        if p == 0:
            return self.C.min_item()[1].mean

        for i, key in enumerate(self.C.keys()):
            c_i_plus_one = self.C[key]
            if i == 0:
                k = c_i_plus_one.count / 2

            else:
                k = (c_i_plus_one.count + c_i.count) / 2.
                if p <= t + k:
                    z1 = p - t
                    z2 = t + k - p
                    return (c_i.mean * z2 + c_i_plus_one.mean * z1) / k
            c_i = c_i_plus_one
            t += k

        return self.C.max_item()[1].mean

    def cdf(self, x):
        """
        Computes the cdf of a specific value, ie. computes F(x) where F denotes
        the CDF of the distribution.
        """
        t = 0
        N = float(self.n)

        if len(self) == 1:  # only one centroid
            return int(x >= self.C.min_key())

        for i, key in enumerate(self.C.keys()):
            c_i = self.C[key]
            if i == len(self) - 1:
                delta = (c_i.mean - self.C.prev_item(key)[1].mean) / 2.
            else:
                delta = (self.C.succ_item(key)[1].mean - c_i.mean) / 2.
            z = max(-1, (x - c_i.mean) / delta)

            if z < 1:
                return t / N + c_i.count / N * (z + 1) / 2

            t += c_i.count
        return 1

    def trimmed_mean(self, p1, p2):
        """
        Computes the mean of the distribution between the two percentiles p1 and p2.
        This is a modified algorithm than the one presented in the original t-Digest paper.

        """
        if not (p1 < p2):
            raise ValueError("p1 must be between 0 and 100 and less than p2.")

        min_count = p1 / 100. * self.n
        max_count = p2 / 100. * self.n

        trimmed_sum = trimmed_count = curr_count = 0
        for i, c in enumerate(self.C.values()):
            next_count = curr_count + c.count
            if next_count <= min_count:
                curr_count = next_count
                continue

            count = c.count
            if curr_count < min_count:
                count = next_count - min_count
            if next_count > max_count:
                count -= next_count - max_count

            trimmed_sum += count * c.mean
            trimmed_count += count

            if next_count >= max_count:
                break
            curr_count = next_count

        if trimmed_count == 0:
            return 0
        return trimmed_sum / trimmed_count

    def centroids_to_list(self):
        """
        Returns a Python list of the TDigest object's Centroid values.

        """
        centroids = []
        for key in self.C.keys():
            tree_values = self.C.get_value(key)
            centroids.append({'m':tree_values.mean, 'c':tree_values.count})
        return centroids

    def to_dict(self):
        """
        Returns a Python dictionary of the TDigest and internal Centroid values.
        Or use centroids_to_list() for a list of only the Centroid values.

        """
        return {'n':self.n, 'delta':self.delta, 'K':self.K, 'centroids':self.centroids_to_list()}

    @classmethod
    def from_dict(cls, d:dict):
       t = TDigest()
       for c in d['centroids']:
           ci = Centroid(c['m'],c['c'])
           t.C.insert(c['m'], ci)
       t.K = d['K']
       t.delta = d['delta']
       t.n = d['n']
       return t 
    
    def update_from_dict(self, dict_values):
        """
        Updates TDigest object with dictionary values.

        The digest delta and K values are optional if you would like to update them,
        but the n value is not required because it is computed from the centroid weights.

        For example, you can initalize a new TDigest:
            digest = TDigest()
        Then load dictionary values into the digest:
            digest.update_from_dict({'K': 25, 'delta': 0.01, 'centroids': [{'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}]})

        Or update an existing digest where the centroids will be appropriately merged:
            digest = TDigest()
            digest.update(1)
            digest.update(2)
            digest.update(3)
            digest.update_from_dict({'K': 25, 'delta': 0.01, 'centroids': [{'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}]})

        Resulting in the digest having merged similar centroids by increasing their weight:
            {'K': 25, 'delta': 0.01, 'centroids': [{'c': 2.0, 'm': 1.0}, {'c': 2.0, 'm': 2.0}, {'c': 2.0, 'm': 3.0}], 'n': 6.0}

        Alternative you can provide only a list of centroid values with update_centroids_from_list()

        """
        self.delta = dict_values.get('delta', self.delta)
        self.K = dict_values.get('K', self.K)
        self.update_centroids_from_list(dict_values['centroids'])
        return self

    def update_centroids_from_list(self, list_values):
        """
        Add or update Centroids from a Python list.
        Any existing centroids in the digest object are appropriately updated.

        Example:
            digest.update_centroids([{'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}])

        """
        [self.update(value['m'], value['c']) for value in list_values]
        return self


if __name__ == '__main__':
    from numpy import random

    T1 = TDigest()
    x = random.random(size=10000)
    T1.batch_update(x)

    print(abs(T1.percentile(50) - 0.5))
    print(abs(T1.percentile(10) - .1))
    print(abs(T1.percentile(90) - 0.9))
    print(abs(T1.percentile(1) - 0.01))
    print(abs(T1.percentile(0.1) - 0.001))
    print(T1.trimmed_mean(0.5, 1.))
