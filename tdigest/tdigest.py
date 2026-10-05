from __future__ import print_function

from bisect import bisect_right
from random import choice
from itertools import chain
from accumulation_tree import AccumulationTree
import pyudorandom


def _centroid_count(centroid):
    # this is defined at the top level because
    # pyspark needs to pickle these functions properly
    # See https://github.com/CamDavidsonPilon/tdigest/issues/42
    return centroid.count


def _is_finite(value):
    return float('-inf') < value < float('inf')


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

        return self.quantile(float(p) / 100.)

    def quantile(self, q):
        """
        Computes the inverse CDF at a probability in [0, 1].
        """
        if not (0 <= q <= 1):
            raise ValueError("q must be between 0 and 1, inclusive.")

        quantiles, values = self.centroid_interpolation()
        index = bisect_right(quantiles, q) - 1

        if index == len(quantiles) - 1:
            return values[-1]

        lower_q = quantiles[index]
        upper_q = quantiles[index + 1]
        lower_value = values[index]
        upper_value = values[index + 1]
        position = (q - lower_q) / (upper_q - lower_q)
        return lower_value + position * (upper_value - lower_value)

    def centroid_interpolation(self):
        """
        Returns the normalized piecewise-linear quantile function.

        The two returned lists contain cumulative-probability breakpoints and
        values. Between two breakpoints the quantile function is linear. The
        first and last centroid each account for half of their weight at their
        centroid value, matching this digest's percentile interpolation.
        """
        centroids = []
        for centroid in self.C.values():
            if not _is_finite(centroid.mean):
                raise ValueError("centroid means must be finite.")
            if not _is_finite(centroid.count):
                raise ValueError("centroid counts must be finite.")
            if centroid.count < 0:
                raise ValueError("centroid counts must be non-negative.")
            if centroid.count > 0:
                centroids.append(centroid)

        if not centroids:
            raise ValueError("Cannot interpolate an empty TDigest.")

        total_count = float(self.n)
        if not _is_finite(total_count) or total_count <= 0:
            raise ValueError("Total centroid weight must be finite and positive.")

        quantiles = [0., centroids[0].count / (2. * total_count)]
        values = [centroids[0].mean, centroids[0].mean]
        cumulative_count = centroids[0].count

        for centroid in centroids[1:]:
            quantiles.append((cumulative_count + centroid.count / 2.) /
                             total_count)
            values.append(centroid.mean)
            cumulative_count += centroid.count

        quantiles.append(1.)
        values.append(centroids[-1].mean)
        return quantiles, values

    def wasserstein_distance(self, other_digest):
        """
        Computes the first-order Wasserstein distance between the two
        normalized quantile functions represented by these digests.

        Each digest is normalized by its own total centroid weight, so digests
        with different total weights can be compared as probability
        distributions. Empty digests and centroids with non-finite means or
        counts are not valid inputs. This is a distance between the
        t-digest summaries' interpolation models, not an exact distance for
        the original observations.
        """
        if not isinstance(other_digest, TDigest):
            raise TypeError("other_digest must be a TDigest object.")

        left_quantiles, left_values = self.centroid_interpolation()
        right_quantiles, right_values = other_digest.centroid_interpolation()

        def value_at(quantiles, values, index, quantile):
            lower_quantile = quantiles[index]
            upper_quantile = quantiles[index + 1]
            position = (quantile - lower_quantile) / (upper_quantile - lower_quantile)
            return values[index] + position * (values[index + 1] - values[index])

        left_index = 0
        right_index = 0
        distance = 0.

        while (left_index < len(left_quantiles) - 1 and
               right_index < len(right_quantiles) - 1):
            lower_quantile = max(left_quantiles[left_index],
                                 right_quantiles[right_index])
            upper_quantile = min(left_quantiles[left_index + 1],
                                 right_quantiles[right_index + 1])

            if upper_quantile > lower_quantile:
                left_lower = value_at(left_quantiles, left_values,
                                      left_index, lower_quantile)
                left_upper = value_at(left_quantiles, left_values,
                                      left_index, upper_quantile)
                right_lower = value_at(right_quantiles, right_values,
                                       right_index, lower_quantile)
                right_upper = value_at(right_quantiles, right_values,
                                       right_index, upper_quantile)

                lower_difference = left_lower - right_lower
                upper_difference = left_upper - right_upper

                if lower_difference * upper_difference < 0:
                    width = upper_quantile - lower_quantile
                    root_position = abs(lower_difference) / (
                        abs(lower_difference) + abs(upper_difference))
                    root_quantile = lower_quantile + width * root_position
                    distance += ((root_quantile - lower_quantile) *
                                 abs(lower_difference) / 2.)
                    distance += ((upper_quantile - root_quantile) *
                                 abs(upper_difference) / 2.)
                else:
                    distance += ((upper_quantile - lower_quantile) *
                                 (abs(lower_difference) +
                                  abs(upper_difference)) / 2.)

            if left_quantiles[left_index + 1] < right_quantiles[right_index + 1]:
                left_index += 1
            elif right_quantiles[right_index + 1] < left_quantiles[left_index + 1]:
                right_index += 1
            else:
                left_index += 1
                right_index += 1

        return distance

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
