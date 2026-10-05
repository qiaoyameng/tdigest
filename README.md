# tdigest
### Efficient percentile estimation of streaming or distributed data
[![PyPI version](https://badge.fury.io/py/tdigest.svg)](https://badge.fury.io/py/tdigest)
[![Build Status](https://travis-ci.org/CamDavidsonPilon/tdigest.svg?branch=master)](https://travis-ci.org/CamDavidsonPilon/tdigest)


This is a Python implementation of Ted Dunning's [t-digest](https://github.com/tdunning/t-digest) data structure. The t-digest data structure is designed around computing accurate estimates from either streaming data, or distributed data. These estimates are percentiles, quantiles, trimmed means, etc. Two t-digests can be added, making the data structure ideal for map-reduce settings, and can be serialized into much less than 10kB (instead of storing the entire list of data).

See a blog post about it here: [Percentile and Quantile Estimation of Big Data: The t-Digest](http://dataorigami.net/blogs/napkin-folding/19055451-percentile-and-quantile-estimation-of-big-data-the-t-digest)


### Installation
*tdigest* is compatible with both Python 2 and Python 3. 

```
pip install tdigest
```

### Usage

#### Update the digest sequentially

```
from tdigest import TDigest
from numpy.random import random

digest = TDigest()
for x in range(5000):
    digest.update(random())

print(digest.percentile(15))  # about 0.15, as 0.15 is the 15th percentile of the Uniform(0,1) distribution
```

#### Update the digest in batches

```
another_digest = TDigest()
another_digest.batch_update(random(5000))
print(another_digest.percentile(15))
```

#### Sum two digests to create a new digest

```
sum_digest = digest + another_digest 
sum_digest.percentile(30)  # about 0.3
```

#### Percentiles, CDF values, and observed extremes

Compression can move the mean of an outer centroid away from the actual
smallest or largest observation. A digest therefore keeps `min` and `max`
separately from its centroids. `percentile(0)` and `percentile(100)` return
those exact observed values after updates, compression, digest merges, and
round-trip serialization.

The internal distribution is a piecewise-linear approximation. Its CDF is
right-continuous, `F(x) = P(X <= x)`, and percentiles use the lower
generalized inverse `Q(p) = inf{x: F(x) >= p}`. Consequently, at a jump the
CDF includes the mass at the jump point and a percentile chooses the left
(lower) side. Centroids are not used to recover original samples.

```
digest.min, digest.max
view = digest.distribution_view()
view.cdf(0.25)
view.percentile(75)
```

#### To dict or serializing a digest with JSON

You can use the to_dict() method to turn a TDigest object into a standard Python dictionary.
```
digest = TDigest()
digest.update(1)
digest.update(2)
digest.update(3)
print(digest.to_dict())
```
Or you can get only a list of Centroids with `centroids_to_list()`.
```
digest.centroids_to_list()
```

Similarly, you can restore a Python dict of digest values with `update_from_dict()`. Centroids are merged with any existing ones in the digest.
For example, make a fresh digest and restore values from a python dictionary.
```
digest = TDigest()
digest.update_from_dict({'K': 25, 'delta': 0.01, 'centroids': [{'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}]})
```
New dictionaries also contain `min` and `max`. Older summaries that omit those
fields have unknown exact endpoints: their `min` and `max` are `None`, and
`percentile(0)`/`percentile(100)` return `None` rather than fabricating
endpoints from outer centroid means. Their interior interpolation remains
available through `cdf()` and non-extreme `percentile()` calls.

K and delta values are optional, or you can provide only a list of centroids with `update_centroids_from_list()`.
```
digest = TDigest()
digest.update_centroids([{'c': 1.0, 'm': 1.0}, {'c': 1.0, 'm': 2.0}, {'c': 1.0, 'm': 3.0}])
```

If you want to serialize with other tools like JSON, you can first convert to_dict().
```
json.dumps(digest.to_dict())
```

Alternatively, make a custom encoder function to provide as default to the standard json module.
```
def encoder(digest_obj):
    return digest_obj.to_dict()
```
Then pass the encoder function as the default parameter.
```
json.dumps(digest, default=encoder)
```


### API 

`TDigest.`

 - `update(x, w=1)`: update the tdigest with value `x` and weight `w`.
 - `batch_update(x, w=1)`: update the tdigest with values in array `x` and weight `w`.
 - `compress()`: perform a compression on the underlying data structure that will shrink the memory footprint of it, without hurting accuracy. Good to perform after adding many values. 
 - `percentile(p)`: return the `p`th percentile. Example: `p=50` is the median.
 - `cdf(x)`: return the CDF the value `x` is at. 
 - `distribution_view()`: return the piecewise-linear approximate distribution shared by `percentile()` and `cdf()`.
 - `min` / `max`: exact observed endpoints, or `None` when imported centroids do not provide them.
 - `trimmed_mean(p1, p2)`: return the mean of data set without the values below and above the `p1` and `p2` percentile respectively. 
 - `to_dict()`: return a Python dictionary of the TDigest and internal Centroid values.
 - `update_from_dict(dict_values)`: update from serialized dictionary values into the TDigest object.
 - `centroids_to_list()`: return a Python list of the TDigest object's internal Centroid values.
 - `update_centroids_from_list(list_values)`: update Centroids from a python list.

 


