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


class DigestDistribution(object):
    """
    A consistent approximate distribution (CDF + quantile function) built from
    a t-digest. This is a *view*: it describes the distribution implied by the
    digest's centroids and observed extremes, it does not recover the original
    samples.

    Interpolation model
    -------------------
    Each centroid (m_i, w_i) contributes the knot (m_i, s_i) where
    s_i = w_i / 2 + sum_{j<i} w_j is the cumulative weight up to the centroid
    midpoint. The CDF F is the piecewise-linear interpolation of these knots,
    rescaled by the total weight n.

    When the exact observed extremes are known, the knots (min, 0) and
    (max, n) are added, so quantile(0) == min and quantile(1) == max exactly
    and no probability mass is placed outside [min, max]. If min (resp. max)
    coincides with the outermost centroid mean, the cdf jumps there from 0
    (resp. to 1): the outer half of that centroid's mass sits at its mean.
    When the extremes
    are unknown (e.g. a legacy serialized summary that never recorded them),
    min_value/max_value are None and no endpoint is fabricated: F is 0 below
    the first centroid mean and 1 above the last one, and quantile(0) /
    quantile(1) fall back to the outermost centroid means as estimates.

    Generalized inverse at jumps
    ----------------------------
    F may jump: repeated values collapse into a single centroid, and a known
    extreme may coincide with the outermost centroid mean. F is taken to be
    right-continuous, and quantile() is the lower generalized inverse

        Q(p) = inf{ x : F(x) >= p }.

    This keeps the two maps mutually compatible: both are non-decreasing and
    for all x in [min, max] and 0 <= p <= 1,

        cdf(quantile(p)) >= p   and   quantile(cdf(x)) <= x.
    """

    def __init__(self, centroids, n, min_value=None, max_value=None):
        self.n = float(n)
        if self.n <= 0:
            raise ValueError("Cannot build a distribution from an empty digest.")
        self.min_value = min_value
        self.max_value = max_value

        # Knots are (x, cumulative weight) pairs; weights are rescaled by n
        # only when reported, to limit floating point error.
        knots = []
        if min_value is not None:
            knots.append((float(min_value), 0.0))
        cumulative = 0.0
        for centroid in centroids:
            knots.append((centroid.mean, cumulative + centroid.count / 2.0))
            cumulative += centroid.count
        if max_value is not None:
            knots.append((float(max_value), self.n))
        self._knots = knots

    def __repr__(self):
        return """<DigestDistribution: n=%d, knots=%d, min=%s, max=%s>""" % (
            self.n, len(self._knots), self.min_value, self.max_value)

    def cdf(self, x):
        """
        Computes F(x), the fraction of the total weight at values <= x.
        """
        knots = self._knots
        if x < knots[0][0]:
            return 0.0
        if x >= knots[-1][0]:
            return 1.0
        for i in range(len(knots) - 1):
            x0, y0 = knots[i]
            x1, y1 = knots[i + 1]
            # Zero-width segments are jumps; F is right-continuous there.
            if x0 <= x < x1:
                return (y0 + (y1 - y0) * (x - x0) / (x1 - x0)) / self.n

    def quantile(self, p):
        """
        Computes the lower generalized inverse Q(p) = inf{ x : F(x) >= p }
        for p in [0, 1].
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0 and 1, inclusive.")
        return self._quantile_at_count(p * self.n)

    def _quantile_at_count(self, c):
        knots = self._knots
        x0, y0 = knots[0]
        if c <= y0:
            return x0
        for i in range(1, len(knots)):
            x1, y1 = knots[i]
            if c <= y1:
                if x1 == x0:
                    # Jump: the lower generalized inverse is the jump location.
                    return x1
                return (x0 * (y1 - c) + x1 * (c - y0)) / (y1 - y0)
            x0, y0 = x1, y1
        return knots[-1][0]


class TDigest(object):

    def __init__(self, delta=0.01, K=25):

        self.C = AccumulationTree(_centroid_count)
        self.n = 0
        self.delta = delta
        self.K = K
        # Exact observed extremes; None means "unknown" (e.g. imported from a
        # legacy summary that never recorded them).
        self._min = None
        self._max = None

    def __add__(self, other_digest):
        data = list(chain(self.C.values(), other_digest.C.values()))
        new_digest = TDigest(self.delta, self.K)

        if len(data) > 0:
            for c in pyudorandom.items(data):
                new_digest.update(c.mean, c.count)

        new_digest._min = self._merge_extreme(self._min, self.n, other_digest._min, other_digest.n, min)
        new_digest._max = self._merge_extreme(self._max, self.n, other_digest._max, other_digest.n, max)
        return new_digest

    @staticmethod
    def _merge_extreme(own, own_n, other, other_n, op):
        # None means "unknown" and propagates: an extreme is only exact if
        # both non-empty operands know theirs.
        if own_n == 0:
            return other
        if other_n == 0:
            return own
        if own is None or other is None:
            return None
        return op(own, other)

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
        if self.n == 0 and self._min is None:
            # First observation on a fresh digest establishes exact extremes.
            self._min = self._max = x
        elif self._min is not None:
            self._min = min(self._min, x)
            self._max = max(self._max, x)
        # else: extremes are unknown (legacy import); leave them unknown
        # rather than fabricating exact endpoints from centroid means.
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
        T._min = self._min
        T._max = self._max
        self.C = T.C

    def distribution(self):
        """
        Returns a DigestDistribution view of this digest: a piecewise-linear
        approximate CDF and its (lower generalized) inverse that preserve the
        exact observed extremes when they are known. percentile() and cdf()
        are both defined in terms of this view.
        """
        return DigestDistribution(self.C.values(), self.n, self._min, self._max)

    def percentile(self, p):
        """
        Computes the percentile of a specific value in [0,100].
        percentile(0) and percentile(100) return the exact observed minimum
        and maximum when they are known. Shares its interpolation model with
        cdf(); see DigestDistribution for the precise convention.
        """
        if not (0 <= p <= 100):
            raise ValueError("p must be between 0 and 100, inclusive.")
        return self.distribution().quantile(float(p) / 100.)

    def cdf(self, x):
        """
        Computes the cdf of a specific value, ie. computes F(x) where F denotes
        the CDF of the distribution. Shares its interpolation model with
        percentile(); see DigestDistribution for the precise convention.
        """
        return self.distribution().cdf(x)

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

        Includes the exact observed extremes under the 'min' and 'max' keys.
        These are None when the extremes are unknown (e.g. the digest was
        imported from a legacy summary that never recorded them).

        """
        return {'n':self.n, 'delta':self.delta, 'K':self.K,
                'min':self._min, 'max':self._max,
                'centroids':self.centroids_to_list()}

    @classmethod
    def from_dict(cls, d:dict):
       t = TDigest()
       for c in d['centroids']:
           ci = Centroid(c['m'],c['c'])
           t.C.insert(c['m'], ci)
       t.K = d['K']
       t.delta = d['delta']
       t.n = d['n']
       # Legacy summaries have no 'min'/'max' keys: the extremes are then
       # unknown (None) rather than fabricated from the outermost centroids.
       t._min = d.get('min')
       t._max = d.get('max')
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
        pre_n = self.n
        pre_min, pre_max = self._min, self._max
        self.update_centroids_from_list(dict_values['centroids'])
        # Merging centroid means must not be mistaken for observing exact
        # extremes, so combine the recorded extremes explicitly. If either
        # side never recorded them (legacy summaries), the merged extremes
        # stay unknown instead of being fabricated.
        other_min = dict_values.get('min')
        other_max = dict_values.get('max')
        other_n = len(dict_values['centroids'])
        if pre_n == 0:
            self._min, self._max = other_min, other_max
        else:
            self._min = self._merge_extreme(pre_min, pre_n, other_min, other_n, min)
            self._max = self._merge_extreme(pre_max, pre_n, other_max, other_n, max)
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
