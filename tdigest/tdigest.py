from __future__ import print_function

import math

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


class EmptyDigestError(Exception):
    """
    Raised when an exponentially decaying digest is queried but has no
    positive effective weight left (all historical samples have decayed to
    zero or the digest has never received an update).
    """
    pass


class ExponentiallyDecayingTDigest(object):
    """
    T-Digest wrapper in which the weight of every sample decays exponentially
    with wall-clock time, with a fixed half-life. Time is always supplied by
    the caller; the wrapper never reads a clock itself.

    A sample inserted at time ``t_i`` with weight ``w_i`` contributes

        w_i * 0.5 ** ((t - t_i) / half_life)

    of *effective weight* at query time ``t``.

    Rather than touching every centroid on each update, the wrapper keeps a
    single global scale factor ``g`` such that the *effective* weight of a
    centroid at ``last_time`` is ``stored_count * g``. Moving forward in time
    only multiplies ``g`` by the decay factor; the centroids themselves are
    untouched. A new sample is inserted with stored weight ``w / g`` so its
    effective weight is exactly ``w``. The TDigest's quantile compression
    thresholds depend only on relative weights, which are unaffected by the
    common factor, so merging and compression behave identically to an
    undecayed digest.

    When ``g`` shrinks below a safety threshold chosen with safe margins
    from both float64 limits, every centroid is multiplied by ``g`` in a
    single renormalization pass and ``g`` resets to 1: this is the only
    point at which centroids are traversed, and it happens at most once per
    few hundred half-lives of update-span. All query methods
    (percentile/cdf/trimmed_mean) are homogeneous of degree zero in the
    weights, so they return identical answers without any rescaling, even
    after arbitrarily long idle periods.

    When the decay factor from ``last_time`` underflows, every old sample
    carries zero effective weight: the digest is reset to an explicit empty
    state instead of keeping numerically dead data (which could otherwise
    surface again after a later renormalization).
    """

    # Renormalize once the global scale g falls below this bound. Chosen far
    # from the float64 underflow (~1e-308) limit so that, after multiplying
    # counts by g during renormalization, no centroid that still has a
    # representable effective weight is lost; the reciprocal (~1e100) is far
    # from the overflow (~1e308) limit for newly inserted stored weights.
    _RENORMALIZE_BELOW = 1e-100

    def __init__(self, half_life, delta=0.01, K=25):
        if not isinstance(half_life, (int, float)) or not math.isfinite(half_life) or half_life <= 0:
            raise ValueError("half_life must be a finite positive number.")

        self.half_life = float(half_life)
        self.delta = delta
        self.K = K
        self._digest = TDigest(delta, K)
        self.last_time = None
        # Effective weight at last_time == stored_count * g.
        self.g = 1.0

    def __repr__(self):
        return """<ExponentiallyDecaying T-Digest: half_life=%g, last_time=%s, effective_weight=%.8g>""" % (
            self.half_life, self.last_time, self.effective_weight())

    def __len__(self):
        return len(self._digest)

    def __iter__(self):
        return iter(self.effective_centroids())

    def is_empty(self, time=None):
        """
        Returns True if the digest holds no samples or their total effective
        weight has decayed to zero. With no argument, reports the state at the
        last update time; with ``time`` it applies the same decay rule as a
        query (and rejects backwards time).
        """
        if self.last_time is None:
            return True
        if time is None:
            return not (self._digest.n > 0)
        return self.effective_weight(time) <= 0

    def effective_weight(self, time=None):
        """
        Total weight of all samples at ``time`` (defaults to the last update
        time), after exponential decay.
        """
        if self.last_time is None:
            return 0.0
        if time is None:
            return float(self._digest.n) * self.g
        self._check_time(time)
        return self.g * self._decay_factor(time) * self._digest.n

    def effective_centroids(self, time=None):
        """
        Returns ``[(mean, effective_count), ...]`` ordered by mean, with every
        centroid weight decayed to ``time`` (defaults to the last update
        time). Useful for tests and diagnostics.
        """
        factor = 1.0
        factor *= self.g
        if time is not None:
            self._check_time(time)
            factor *= self._decay_factor(time)
        return [(c.mean, c.count * factor)
                for c in self._digest.C.values()
                if c.count * factor > 0]

    def _check_time(self, time):
        if not isinstance(time, (int, float)) or not math.isfinite(time):
            raise ValueError("time must be a finite number.")
        if self.last_time is not None and time < self.last_time:
            raise ValueError(
                "time must be non-decreasing: got %r after last_time %r"
                % (time, self.last_time))

    def _decay_factor(self, time):
        """Multiplicative weight factor from ``last_time`` to ``time``."""
        return 0.5 ** (float(time - self.last_time) / self.half_life)

    def _reset(self, time):
        self._digest = TDigest(self.delta, self.K)
        self.last_time = time
        self.g = 1.0

    def _renormalize(self):
        """
        Rewrite all stored centroid weights onto the current scale
        (count *= g) and reset g to 1. This is the only operation that
        traverses the centroids; it runs at most once per
        -log2(_RENORMALIZE_BELOW) ~= 332 half-lives of update-span.
        Centroids whose rescaled count is exactly zero or non-finite are
        dropped, and the total is recomputed from the survivors.
        """
        scale = self.g
        new_n = 0.0
        dead_keys = []
        for key, centroid in self._digest.C.items():
            new_count = centroid.count * scale
            if new_count > 0 and math.isfinite(new_count):
                centroid.count = new_count
                new_n += new_count
            else:
                dead_keys.append(key)
        for key in dead_keys:
            self._digest.C.pop(key)
        self._digest.n = new_n
        self.g = 1.0

    def update(self, x, time, w=1):
        """
        Add value ``x`` with weight ``w`` at caller-supplied ``time``.

        Times must be non-decreasing; going backwards raises ValueError.
        """
        self._check_time(time)
        if not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0:
            raise ValueError("w must be a finite positive number.")
        if not math.isfinite(float(x)):
            raise ValueError("x must be finite.")

        if self.last_time is None:
            self._reset(time)
            self._digest.update(x, w)
            return

        factor = self._decay_factor(time)

        # Everything old has decayed away: discard it explicitly rather than
        # carrying (and possibly later "reviving") numerically dead weights.
        new_g = self.g * factor
        if not math.isfinite(factor) or not (new_g * self._digest.n > 0):
            self._reset(time)
            self._digest.update(x, w)
            return

        self.g = new_g

        if self.g < self._RENORMALIZE_BELOW:
            self._renormalize()

        # The new sample's effective weight at time is w; its stored weight
        # on the current scale (g) is therefore w / g.
        self._digest.update(x, w / self.g)
        self.last_time = time

    def batch_update(self, values, time, w=1):
        """
        Update with an iterable of values all observed at the same ``time``.
        """
        self._check_time(time)
        for x in values:
            self.update(x, time, w)
        self._digest.compress()

    def _query(self, time):
        if self.is_empty():
            raise EmptyDigestError(
                "Digest is empty: no samples or all weights have decayed to zero.")
        if time is not None:
            if self.is_empty(time):
                raise EmptyDigestError(
                    "All weights have decayed to zero at time %r." % (time,))
        return self._digest

    def percentile(self, p, time=None):
        """
        Percentile in [0, 100] at ``time`` (defaults to the last update
        time). Scale-invariant, so long idle gaps need no rescaling.
        """
        return self._query(time).percentile(p)

    def cdf(self, x, time=None):
        """Cumulative distribution at ``x`` and ``time``."""
        return self._query(time).cdf(x)

    def trimmed_mean(self, p1, p2, time=None):
        """Weighted trimmed mean between percentiles ``p1`` and ``p2``."""
        return self._query(time).trimmed_mean(p1, p2)


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
