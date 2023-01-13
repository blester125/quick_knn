"""Hash Family for Cosine Distance via random pooling.

Based on: http://personal.denison.edu/~lalla/papers/online-lsh.pdf
"""

import struct
import numpy as np

from quick_knn.hashes import HashFamily
from quick_knn.hashes.random_hyperplanes import RandomHyperplanesBase


MAX = (1 << 64) - 1


class PoolingRandomHyperPlanes(RandomHyperplanesBase):
    def __init__(self, signature_size: int, pool_size: int, seed: int):
        super().__init__(signature_size, seed)
        self._pool_size = pool_size
        rng = np.random.RandomState(seed)
        # For each single hyperplane hash, use a unique offset to get different
        # bucket distributions.
        self.hash_offsets = rng.randint(
            0, MAX, size=self.signature_size, dtype=np.uint64
        )
        self.pool = rng.normal(size=self.pool_size).astype(dtype=np.float32)

    @property
    def pool_size(self):
        return self._pool_size

    def hash(self, x):
        signature = np.zeros(self.signature_size, dtype=np.uint8)
        # TODO: Can these loops be converted to Cython/Numba to make it faster?
        for i in range(self.signature_size):
            dot = np.sum(
                [
                    self.pool[(j ^ self.hash_offsets[i]) % self.pool_size] * f
                    for j, f in enumerate(x)
                ]
            )
            signature[i] = 1 if dot >= 0 else 0
        return signature

    def __repr__(self):
        return f"{self.__class__.__name__}(distance={self.name}, signature_size={self.signature_size}, pool_size={self.pool_size}, seed={self.seed}, sensitivity={self.sensitivity})"


if __name__ == "__main__":
    prhp = PoolingRandomHyperPlanes(32, 10_000, 0)
    print(prhp.name)
    print(prhp.sensitivity)
    print(prhp.signature_size)
    print(prhp.pool_size)

    r = np.random.RandomState(12)
    x = r.rand(100)

    print(x)
    print(prhp.hash(x))
