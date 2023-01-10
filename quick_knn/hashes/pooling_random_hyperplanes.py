"""Hash Family for Cosine Distance via random pooling.

Based on: http://personal.denison.edu/~lalla/papers/online-lsh.pdf
"""

import struct
import numpy as np

from quick_knn.hashes import HashFamily
from quick_knn.hashes.random_hyperplanes import RandomHyperplanesBase


MAX = (1 << 64) - 1


def float_to_bits(f: float) -> int:
    """Convert float to the bit representation for a 'hash'."""
    # Neither struct or ctypes is available in numba
    # Enforce big endian to be consistent across computers.
    # return ctypes.c_uint64.from_buffer(ctypes.c_double(f)).value
    # Pack the float into bytes and then unpack into a long. Like the pointer
    # cast from Quake fast inverse sqrt.
    # TODO: Does a using a float32 like this cause issues with hashing? Are only
    # half of our buckets used?
    bits = struct.pack(">f", f)
    return struct.unpack(">l", bits)[0]


def pool_hash(feature, offset: int) -> int:
    """A hash function that mixes the feature and a particular hash family member."""
    return float_to_bits(feature) ^ int(offset)


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
                    self.pool[pool_hash(f, self.hash_offsets[i]) % self.pool_size]
                    for f in x
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
