from __future__ import print_function

from math import isfinite
from random import choice
from itertools import chain
from accumulation_tree import AccumulationTree
import pyudorandom


def _centroid_count(centroid):
    # this is defined at the top level because
    # pyspark needs to pickle these functions properly
    # See https://github.com/CamDavidsonPilon/tdigest/issues/42
    return centroid.count


def _cdf_at_breakpoint(points, x, side='right'):
    if not isfinite(x):
        raise ValueError("cdf evaluation point must be finite")
    if x < points[0][0]:
        return 0.
    if x > points[-1][0]:
        return 1.

    for i, (mean, cdf_left, cdf_right) in enumerate(points):
        if x == mean:
            return cdf_right if side == 'right' else cdf_left
        if x < mean:
            previous_mean, _, previous_cdf = points[i - 1]
            width = mean - previous_mean
            return previous_cdf + (cdf_left - previous_cdf) * (x - previous_mean) / width

    return 1.


def _linear_absolute_area(width, first, second):
    if first * second < 0:
        return width / 2. * (first * first + second * second) / abs(first - second)
    return width / 2. * (abs(first) + abs(second))


def wasserstein_distance(first_digest, second_digest):
    """
    Compute the first Wasserstein distance between two TDigest summaries.

    The distance is between the normalized, piecewise-linear distributions
    represented by the centroid summaries, not an exact distance between the
    original observations. Each digest is normalized by its own finite,
    positive total centroid weight, so unequal totals are supported. Empty
    digests, non-finite values, and non-positive weights raise ValueError.
    """
    first_points = first_digest.centroid_distribution()
    second_points = second_digest.centroid_distribution()
    breakpoints = sorted(set(
        [point[0] for point in first_points] +
        [point[0] for point in second_points]
    ))

    distance = 0.
    for left, right in zip(breakpoints, breakpoints[1:]):
        first_at_left = _cdf_at_breakpoint(first_points, left, 'right')
        second_at_left = _cdf_at_breakpoint(second_points, left, 'right')
        first_at_right = _cdf_at_breakpoint(first_points, right, 'left')
        second_at_right = _cdf_at_breakpoint(second_points, right, 'left')

        distance += _linear_absolute_area(
            right - left,
            first_at_left - second_at_left,
            first_at_right - second_at_right
        )
    return distance


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

        q = float(p) / 100.
        centroids, total_weight, anchors = self._centroid_distribution_anchors()

        if len(centroids) == 1 or q * total_weight <= anchors[0]:
            return centroids[0].mean
        if q >= 1:
            return centroids[-1].mean

        target_weight = q * total_weight

        for i in range(len(centroids) - 1):
            left_weight = anchors[i]
            right_weight = anchors[i + 1]
            if target_weight <= right_weight:
                z1 = target_weight - left_weight
                z2 = right_weight - target_weight
                segment_weight = right_weight - left_weight
                return (centroids[i].mean * z2 + centroids[i + 1].mean * z1) / segment_weight

        return centroids[-1].mean

    def cdf(self, x):
        """
        Computes the cdf of a specific value, ie. computes F(x) where F denotes
        the CDF of the distribution. Non-finite values raise ValueError.
        """
        points = self.centroid_distribution()
        return _cdf_at_breakpoint(points, float(x), 'right')

    def centroid_distribution(self):
        """
        Return the piecewise-linear CDF implied by the centroids.

        Each item is ``(mean, cdf_left, cdf_right)``. A centroid's weight is
        split equally between the two neighbouring interpolation intervals;
        half of an endpoint centroid's weight is represented as an endpoint
        jump. CDF values are normalized by the centroid weights, so digests
        with different total weights can be compared as probability
        distributions.
        """
        centroids, total_weight, raw_anchors = self._centroid_distribution_anchors()

        if len(centroids) == 1:
            mean = centroids[0].mean
            return [(mean, 0., 1.)]

        points = []
        for i, centroid in enumerate(centroids):
            anchor = raw_anchors[i] / total_weight
            if i == 0:
                points.append((centroid.mean, 0., anchor))
            elif i == len(centroids) - 1:
                points.append((centroid.mean, anchor, 1.))
            else:
                points.append((centroid.mean, anchor, anchor))
        return points

    def _centroid_distribution_anchors(self):
        centroids = list(self.C.values())
        if not centroids:
            raise ValueError("cannot construct a distribution from an empty TDigest")

        total_weight = 0.
        for centroid in centroids:
            mean = float(centroid.mean)
            count = float(centroid.count)
            if not isfinite(mean) or not isfinite(count) or count <= 0:
                raise ValueError("centroid means and weights must be finite and positive")
            total_weight += count

        if not isfinite(total_weight) or total_weight <= 0:
            raise ValueError("total centroid weight must be finite and positive")

        anchors = []
        cumulative_weight = 0.
        for centroid in centroids:
            anchors.append(cumulative_weight + centroid.count / 2.)
            cumulative_weight += centroid.count

        return centroids, total_weight, anchors

    def wasserstein_distance(self, other_digest):
        """
        First Wasserstein distance between the centroid-interpolated summaries.
        """
        return wasserstein_distance(self, other_digest)

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
