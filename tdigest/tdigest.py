from __future__ import print_function

from random import choice
from itertools import chain
from accumulation_tree import AccumulationTree
import pyudorandom


def _centroid_count(centroid):
    # this is defined at the top level because
    # pyspark needs to pickle these functions properly
    # See https://github.com/CamDavidsonPilon/tdigest/issues/42
    return centroid.count


class DistributionView(object):
    """Piecewise-linear distribution represented by a digest.

    Observed minimum and maximum values are kept as exact q=0 and q=1
    anchors.  Centroids supply the internal q=(cumulative + weight/2)/n
    anchors.  Equal x-values at adjacent anchors form a jump.  The CDF is
    right-continuous and percentiles use the lower generalized inverse
    ``Q(p) = inf{x: F(x) >= p}``.
    """

    def __init__(self, centroids, n, minimum=None, maximum=None):
        merged_centroids = {}
        for mean, count in centroids:
            mean = float(mean)
            count = float(count)
            if count <= 0:
                raise ValueError("centroid counts must be positive.")
            merged_centroids[mean] = merged_centroids.get(mean, 0.0) + count

        self.centroids = [
            (mean, merged_centroids[mean])
            for mean in sorted(merged_centroids)
        ]
        if not self.centroids:
            raise ValueError("distribution view requires at least one centroid.")
        first_mean = self.centroids[0][0]
        last_mean = self.centroids[-1][0]

        self.n = float(n)
        if self.n <= 0:
            raise ValueError("distribution view requires positive total weight.")

        self.min = None if minimum is None else float(minimum)
        self.max = None if maximum is None else float(maximum)

        if self.min is not None and self.min > first_mean:
            raise ValueError("minimum cannot exceed the first centroid mean.")
        if self.max is not None and self.max < last_mean:
            raise ValueError("maximum cannot precede the last centroid mean.")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("minimum cannot exceed maximum.")

    minimum = property(lambda self: self.min)
    maximum = property(lambda self: self.max)

    @staticmethod
    def _endpoint_value(data, *keys):
        for key in keys:
            if key in data:
                value = data[key]
                return None if value is None else float(value)
        return None

    def _groups(self):
        points = []
        if self.min is not None:
            points.append((self.min, 0.0))

        cumulative = 0.0
        for mean, count in self.centroids:
            quantile = (cumulative + count / 2.0) / self.n
            points.append((mean, quantile))
            cumulative += count

        if self.max is not None:
            points.append((self.max, 1.0))

        first_mean = self.centroids[0][0]
        last_mean = self.centroids[-1][0]
        if len(self.centroids) == 1:
            if self.min is None:
                points.append((first_mean, 0.0))
            if self.max is None:
                points.append((last_mean, 1.0))
        else:
            second_mean = self.centroids[1][0]
            previous_mean = self.centroids[-2][0]
            if self.min is None:
                points.append((first_mean - (second_mean - first_mean) / 2.0, 0.0))
            if self.max is None:
                points.append((2.0 * last_mean - previous_mean, 1.0))

        points.sort(key=lambda point: (point[0], point[1]))

        groups = []
        for x, quantile in points:
            if groups and groups[-1][0] == x:
                old_x, low, high = groups[-1]
                groups[-1] = (old_x, min(low, quantile), max(high, quantile))
            else:
                groups.append((x, quantile, quantile))
        return groups

    def percentile(self, p):
        """Return the lower generalized inverse percentile in [0, 100]."""
        p = float(p)
        if not (0.0 <= p <= 100.0):
            raise ValueError("p must be between 0 and 100, inclusive.")

        quantile = p / 100.0
        if quantile == 0.0 and self.min is None:
            return None
        if quantile == 1.0 and self.max is None:
            return None

        groups = self._groups()
        previous_high = 0.0
        for i, (x, low, high) in enumerate(groups):
            if quantile < low:
                previous_x, _, previous_high = groups[i - 1]
                fraction = (quantile - previous_high) / (low - previous_high)
                return previous_x + (x - previous_x) * fraction
            if quantile <= high:
                return x
            previous_high = high
        return groups[-1][0]

    def cdf(self, x):
        """Return the right-continuous CDF F(x) = P(X <= x)."""
        x = float(x)
        groups = self._groups()

        if x < groups[0][0]:
            return 0.0

        for i, (left_x, _, left_high) in enumerate(groups):
            if x <= left_x:
                return left_high
            if i + 1 < len(groups):
                right_x, right_low, _ = groups[i + 1]
                if x < right_x:
                    fraction = (x - left_x) / (right_x - left_x)
                    return left_high + (right_low - left_high) * fraction
        return 1.0


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
        self.min = None
        self.max = None

    minimum = property(lambda self: self.min)
    maximum = property(lambda self: self.max)

    def __add__(self, other_digest):
        return self._merge_with(other_digest)

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

    def _merge_extremes(self, other_digest):
        if self.n == 0:
            return other_digest.min, other_digest.max
        if other_digest.n == 0:
            return self.min, self.max
        if self.min is None or other_digest.min is None:
            minimum = None
        else:
            minimum = min(self.min, other_digest.min)
        if self.max is None or other_digest.max is None:
            maximum = None
        else:
            maximum = max(self.max, other_digest.max)
        return minimum, maximum

    def _merge_with(self, other_digest, in_place=False):
        data = list(chain(self.C.values(), other_digest.C.values()))
        merged = TDigest(self.delta, self.K)

        for centroid in pyudorandom.items(data):
            merged.update(centroid.mean, centroid.count)
        merged.min, merged.max = self._merge_extremes(other_digest)

        if in_place:
            self.C = merged.C
            self.n = merged.n
            self.min = merged.min
            self.max = merged.max
            return self
        return merged

    def update(self, x, w=1):
        """
        Update the t-digest with value x and weight w.

        """
        x = float(x)
        w = float(w)
        if w <= 0:
            raise ValueError("w must be positive.")

        self.n += w
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x

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
        T.min = self.min
        T.max = self.max
        self.C = T.C
        self.n = T.n

    def distribution_view(self):
        """Return the approximate distribution with exact observed extremes."""
        if self.n == 0:
            raise ValueError("cannot create a distribution view from an empty digest.")
        return DistributionView(
            ((centroid.mean, centroid.count) for centroid in self.C.values()),
            self.n,
            self.min,
            self.max,
        )

    def percentile(self, p):
        """
        Computes the percentile of a specific value in [0,100].
        """

        if not (0 <= p <= 100):
            raise ValueError("p must be between 0 and 100, inclusive.")

        if self.n == 0:
            raise ValueError("cannot compute a percentile from an empty digest.")
        return self.distribution_view().percentile(p)

    def cdf(self, x):
        """
        Computes the cdf of a specific value, ie. computes F(x) where F denotes
        the CDF of the distribution.
        """
        if self.n == 0:
            raise ValueError("cannot compute a CDF from an empty digest.")
        return self.distribution_view().cdf(x)

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
        return {
            'n': self.n,
            'delta': self.delta,
            'K': self.K,
            'min': self.min,
            'max': self.max,
            'centroids': self.centroids_to_list(),
        }

    @classmethod
    def from_dict(cls, d:dict):
       t = cls(delta=d.get('delta', 0.01), K=d.get('K', 25))
       total = 0.0
       for c in d.get('centroids', []):
           centroid = Centroid(c['m'], c['c'])
           t._add_centroid(centroid)
           total += centroid.count
       t.n = float(d.get('n', total))
       t.min = DistributionView._endpoint_value(d, 'min', 'minimum')
       t.max = DistributionView._endpoint_value(d, 'max', 'maximum')
       return t

    def update_from_dict(self, d):
        """Merge a serialized digest without treating its centroids as raw data."""
        other = TDigest.from_dict(d)
        self.delta = other.delta
        self.K = other.K
        return self._merge_with(other, in_place=True)

    def update_centroids_from_list(self, centroids):
        """Merge centroids whose exact observed extremes are unknown."""
        other = TDigest(self.delta, self.K)
        for centroid in centroids:
            other.update(centroid['m'], centroid['c'])
        other.min = None
        other.max = None
        return self._merge_with(other, in_place=True)

    update_centroids = update_centroids_from_list


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
