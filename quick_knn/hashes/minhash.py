"""Hash family for Jaccard Distance."""


import dataclasses
import struct
from hashlib import sha1
import numpy as np

from quick_knn.hashes import HashFamily
from quick_knn.utils import Sensitivity


PRIME = (1 << 61) - 1
MAX = (1 << 32) - 1
BYTES = 4


class MinHash(HashFamily):
    def __init__(self, signature_size: int, seed: int = 1, token_hash=sha1):
        super().__init__(signature_size, seed)
        self.inital = np.full(self.signature_size, MAX, dtype=np.uint64)
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, PRIME, dtype=np.uint64, size=self.signature_size)
        self.b = rng.randint(0, PRIME, dtype=np.uint64, size=self.signature_size)
        self.token_hash = token_hash

    @property
    def name(self) -> str:
        return "jaccard"

    @property
    def sensitivity(self) -> Sensitivity:
        return Sensitivity(
            d1="d1",
            d2="d2",
            p1="1 - d1",
            p2="1 - d2",
            p1_func=lambda d1: 1 - d1,
            p2_func=lambda d2: 1 - d2,
        )

    def _hash(self, b: bytes):
        # Instead of converting strings to a vocabulary of integers to build the
        # set contains table, hash them into a bucket representing the column
        # number.
        hashed_token = struct.unpack("<I", self.token_hash(b).digest()[:BYTES])[0]
        # Permute the columns randomly.
        return np.bitwise_and((self.a * hashed_token + self.b) % PRIME, np.uint64(MAX))

    def hash(self, x):
        sig = self.initial
        for token in x:
            # Record the minimum column contained so far.
            sig = np.minimum(self._hash(token), sig)
        return sig

    def similarity(self, query, data) -> float:
        """Jaccard similarity based on MinHash signatures."""
        return np.sum(query == data) / len(query)

    def __repr__(self):
        return f"{self.__class__.__name__}(distance={self.name}, signature_size={self.signature_size}, seed={self.seed}, sensitivity={self.sensitivity})"


if __name__ == "__main__":
    mh = MinHash(32)
    print(mh.sensitivity)
    print(mh)
