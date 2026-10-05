"""
Exponentially time-decaying t-digest.

Monitoring use cases often care most about recent latency, but a hard
cut-off window throws data away abruptly. ``ExponentialDecayingTDigest``
wraps :class:`tdigest.TDigest` and lets every sample's weight decay
exponentially with a fixed half-life. Time is always supplied by the
caller, which keeps the digest deterministic and easy to test.

Decay is applied lazily: instead of rescaling every centroid on each
update, the wrapper tracks a single global scale factor. Stored centroid
weights are the true (decayed) weights divided by that factor, so old
samples lose weight relative to new ones without touching the tree.
When the factor becomes so small that dividing by it risks floating
point overflow, the centroids are renormalized once (all counts
multiplied by the same factor), which preserves relative weights and
keeps the compression thresholds - which scale linearly with the total
weight - consistent.

If the total effective weight decays below ``min_effective_weight``
(including floating point underflow of the decay factor after a very
long idle period), the digest is reset to an explicit empty state:
queries return ``None`` and old data can never resurface.
"""

from .tdigest import TDigest, Centroid


class ExponentialDecayingTDigest(object):

    # When the global scale drops below this value the stored centroid
    # weights are rescaled and the factor is reset to 1.0. This happens
    # at most once per ~332 half-lives, so updates stay O(log n).
    RENORMALIZE_THRESHOLD = 1e-100

    def __init__(self, half_life, delta=0.01, K=25, min_effective_weight=1e-12):
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        if min_effective_weight <= 0:
            raise ValueError("min_effective_weight must be positive")
        self.half_life = float(half_life)
        self.min_effective_weight = float(min_effective_weight)
        self.delta = delta
        self.K = K
        self._digest = TDigest(delta=delta, K=K)
        # true_weight = stored_weight * self._scale
        self._scale = 1.0
        self._last_time = None

    def __repr__(self):
        return "<ExponentialDecayingTDigest: half_life=%g, effective_n=%g, centroids=%d>" % (
            self.half_life, self.effective_count, len(self._digest))

    def __len__(self):
        return len(self._digest)

    @property
    def last_time(self):
        """The most recent timestamp seen, or None before any update."""
        return self._last_time

    @property
    def effective_count(self):
        """Total decayed weight at the current time."""
        return self._digest.n * self._scale

    @property
    def is_empty(self):
        """True when the effective weight has decayed to (numerical) zero."""
        return not (self.effective_count >= self.min_effective_weight)

    def _decay_factor(self, dt):
        # 0.5 ** x underflows cleanly to 0.0 for x >~ 1074, which is
        # handled explicitly in _advance_time.
        return 0.5 ** (dt / self.half_life)

    def _reset(self):
        self._digest = TDigest(delta=self.delta, K=self.K)
        self._scale = 1.0

    def _renormalize(self):
        # Multiply every stored weight by the global scale exactly once,
        # preserving relative weights and the ratio between centroid
        # counts and the compression thresholds (both scale with n).
        digest = self._digest
        fresh = TDigest(delta=self.delta, K=self.K)
        for key in digest.C.keys():
            c = digest.C[key]
            fresh.C.insert(c.mean, Centroid(c.mean, c.count * self._scale))
        fresh.n = digest.n * self._scale
        self._digest = fresh
        self._scale = 1.0

    def _advance_time(self, t):
        if self._last_time is None:
            self._last_time = t
            return
        dt = t - self._last_time
        if dt < 0:
            raise ValueError(
                "time must be monotonically non-decreasing "
                "(last time %r, got %r)" % (self._last_time, t))
        if dt == 0:
            return
        self._last_time = t
        if self._digest.n == 0:
            return
        factor = self._decay_factor(dt)
        if factor == 0.0 or self.effective_count * factor < self.min_effective_weight:
            # Everything stored has decayed to (numerical) zero. Reset
            # so old data cannot be resurrected by later updates.
            self._reset()
            return
        self._scale *= factor
        if self._scale < self.RENORMALIZE_THRESHOLD:
            self._renormalize()

    def update(self, x, t, w=1):
        """
        Update the digest with value ``x`` observed at time ``t`` and
        weight ``w``. ``t`` must not be earlier than any previously
        supplied time.
        """
        if w <= 0:
            raise ValueError("w must be positive")
        self._advance_time(t)
        self._digest.update(x, w / self._scale)
        return self

    def batch_update(self, values, t, w=1):
        """Update with an iterable of values, all observed at time ``t``."""
        for x in values:
            self.update(x, t, w=w)
        self._digest.compress()
        return self

    def percentile(self, p, t=None):
        """
        The ``p``th percentile, or None if the digest is empty. Passing
        ``t`` first decays the digest to that point in time.
        """
        if t is not None:
            self._advance_time(t)
        if self.is_empty:
            return None
        return self._digest.percentile(p)

    def cdf(self, x, t=None):
        """The CDF at ``x``, or None if the digest is empty."""
        if t is not None:
            self._advance_time(t)
        if self.is_empty:
            return None
        return self._digest.cdf(x)

    def trimmed_mean(self, p1, p2, t=None):
        """Trimmed mean between percentiles, or None if the digest is empty."""
        if t is not None:
            self._advance_time(t)
        if self.is_empty:
            return None
        return self._digest.trimmed_mean(p1, p2)
