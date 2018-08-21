from functools import partial
from typing import Tuple, List
from collections import defaultdict
import numpy as np
from quick_knn.data import SQLData, PickleData, sniff
from quick_knn.type_hints import Signature, Integrable, Key, Vector


def integrate(func: Integrable, a: float, b: float, dt: float=0.001) -> float:
    """Midpoint Riemann Sum Integration."""
    area = 0.0
    while a < b:
        area += func(a + 0.5 * dt) * dt
        a += dt
    return area

def fp_prob(b: Vector, r: Vector, s: float) -> Vector:
    # s is Jaccard similarity of two sets
    # s is the probability of a minhash agreeing
    # s^r is the prob of matching at each row in a band of size r
    # (1 - s^r) is the prob of disagreeing in a single row in a band
    # (1 - s^r)^b is the prob of having a mismatch in each band
    # 1 - (1 - s^r)^b is the prob of matching in at least one band
    return 1 - (1 - s ** r) ** b

def fn_prob(b: Vector, r: Vector, s: float) -> Vector:
    # As above (1 - s^r)^b is the probability of having a mismatch in each band.
    return (1 - s ** r) ** b

def opt_b_r(thresh: float, bits: int, fp_weight: float, fn_weight: float) -> Tuple[int, int]:
    """Find optimal values for b and r.

    This calculation is based on:
        * The number of bits in the signature
        * The similarity threshold
        * User weighting for false positives and negatives.
    """
    bs = []
    rs = []
    # Enumerate all legal b and r combinations.
    for b in range(bits):
        max_r = bits // (b + 1)
        for r in range(max_r):
            bs.append(b + 1)
            rs.append(r + 1)
    bs = np.array(bs)
    rs = np.array(rs)
    # Calculate the probability of false positives.
    fp = partial(fp_prob, bs, rs)
    # Integrate from 0 to thresh to calculate the probability of a set with less
    # than the threshold will match in some bucket.
    fp = integrate(fp, 0.0, thresh)
    # Calculate the probability of false positives.
    fn = partial(fn_prob, bs, rs)
    # Integrate from thresh to 1 to calculate the probability of a set with a
    # score greater than the threshold will not match in any bucket.
    fn = integrate(fn, thresh, 1.0)
    # Scale errors by user weighting
    error = fp * fp_weight + fn * fn_weight
    # Pick b and r that minimize the error
    idx = np.argmin(error)
    return bs[idx], rs[idx]


class LSH(object):
    """Banding based LSH as described in http://www.mmds.org/"""

    def __init__(self, threshold: float=0.51, bits: int=32, fp_weight: float=0.5, name="lsh", t="pickle"):
        super().__init__()

        assert threshold <= 1.0 and threshold >= 0.0, f"threshold must be in [0.0, 1.0], got {threshold}"
        self.threshold = threshold
        assert bits >= 2, f"There must be more than 1 bits in your hash, got {bits}"
        self.bits = bits
        assert fp_weight <= 1.0 and fp_weight >= 0.0, f"fp_weight must be in [0.0, 1.0], got {fp_weight}"
        self.fp_weight = fp_weight
        self.fn_weight = 1.0 - fp_weight

        self.b, self.r = opt_b_r(threshold, bits, self.fp_weight, self.fn_weight)

        self.ranges = [(i * self.r, j * self.r) for i, j in zip(range(self.b), range(1, self.b + 1))]
        if t == "pickle":
            self.data = PickleData(name, self.b)
        else:
            self.data = SQLData(name, in_memory=True)

    def __repr__(self):
        return (
            f"LSH(threshold={self.threshold}, bits={self.bits}, "
            f"fp_weight={self.fp_weight:.2}, fn_weight={self.fn_weight:.2}, "
            f"b={self.b}, r={self.r})"
        )

    def insert(self, key: Key, sig: Signature) -> None:
        parts = [LSH.hashable(sig[start:end]) for start, end in self.ranges]
        self.data.insert(parts, key)

    def query(self, sig: Signature) -> List[Key]:
        parts = [LSH.hashable(sig[start:end]) for start, end in self.ranges]
        cands = self.data.get(parts)
        return list(cands)

    @staticmethod
    def hashable(hs: Signature) -> bytes:
        return bytes(hs.data)

    def save(self, hasher):
        self.data.save(self, hasher)

    @classmethod
    def restore(cls, name):
        data = sniff(name)
        return data.restore(name)
