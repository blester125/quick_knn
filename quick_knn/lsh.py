"""Combine multiple hashes in a family to make an LSH."""

from collections import defaultdict
from typing import Optional, Union, Any, List, Set
from quick_knn import banding
from quick_knn.utils import Sensitivity
from quick_knn.hashes import HashFamily
from quick_knn.types import Signature


class LSH:
    def __init__(self, hashes: HashFamily, r: int, b: int):
        self.hashes = hashes
        self.r = r
        self.b = b
        self.buckets = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.data = defaultdict(set)

    def __repr__(self):
        return f"{self.__class__.__name__}(hashes={self.hashes!r}, r={self.r}, b={self.b}, sensitivity={self.sensitivity})"

    @property
    def signature_size(self) -> int:
        return self.hashes.signature_size

    @property
    def sensitivity(self) -> Sensitivity:
        return banding.AndOrSensitivity(
            d1=self.hashes.sensitivity.d1,
            d2=self.hashes.sensitivity.d2,
            p1=self.hashes.sensitivity.p1,
            p2=self.hashes.sensitivity.p2,
            r=self.r,
            b=self.b,
        )

    @staticmethod
    def hashable(signature) -> bytes:
        return bytes(signature.data)

    def band(self, signature: Signature) -> List[bytes]:
        return [LSH.hashable(signature[s:e]) for s, e in self.buckets]

    def insert(self, signature: Signature, item: Any):
        bands = self.band(signature)
        for band in bands:
            self.data[band].add(item)

    def query(self, signature: Signature) -> Set[Any]:
        bands = self.band(signature)
        matches = set()
        for band in bands:
            matches.update(self.data[band])

    def equal(
        self,
        signature_1: Union[Signature, Any],
        signature_2: Union[Signature, Any],
        hash: bool = False,
    ) -> bool:
        if hash:
            signature_1 = self.hashes.hash(signature_1)
            signature_2 = self.hashes.hash(signature_2)
        bands1 = self.band(sig1)
        bands2 = self.band(sig2)
        for b1, b2 in zip(bands1, bands2):
            # Whole band matches
            if b1 == b2:
                # Any number of bands match -> match
                return True
        return False


def smart_settings(
    hashes: HashFamily, threshold: float, false_positive_weight: float
) -> LSH:
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got: {threshold}")
    if not 0.0 < false_positive_weight < 1.0:
        raise ValueError(
            f"false_positive_weight must be in (0, 1), got: {false_positive_weight}"
        )

    r, b = banding.optimize_r_and_b(
        threshold, hashes.signature_size, false_positve_weight
    )

    return LSH(hashes, r, b)


if __name__ == "__main__":
    from quick_knn.hashes.pooling_random_hyperplanes import PoolingRandomHyperPlanes

    lsh = LSH(PoolingRandomHyperPlanes(32, 10000, 0), 8, 4)
    print(lsh)
    print(lsh.sensitivity)
    print(lsh.sensitivity.evaluate(0.2, 0.8, 8, 4))
