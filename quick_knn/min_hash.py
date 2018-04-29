import struct
from hashlib import sha1
import numpy as np

PRIME = (1 << 61) - 1
MAX = (1 << 32) - 1
BYTES = 4


class MinHash(object):

    def __init__(self, bits, seed=1):
        self.bits = bits
        self.default = np.ones(self.bits, dtype=np.uint64) * MAX
        r = np.random.RandomState(seed)
        self.a = r.randint(1, PRIME, dtype=np.uint64, size=self.bits)
        self.b = r.randint(0, PRIME, dtype=np.uint64, size=self.bits)

    def __call__(self, bs, old=None):
        if isinstance(bs, str):
            bs = bs.encode()
            return self.signature(bs, old)
        if isinstance(bs, bytes):
            return self.signature(bs, old)
        for b in bs:
            if isinstance(b, str):
                b = b.encode()
            old = self.signature(b, old)
        return old

    def _hash(self, b):
        hash_values = struct.unpack('<I', sha1(b).digest()[:BYTES])[0]
        perm_hash_values = np.bitwise_and((self.a * hash_values + self.b) % PRIME, np.uint64(MAX))
        return perm_hash_values

    def signature(self, b, old=None):
        if old is None:
            old = self.default
        return np.minimum(self._hash(b), old)

    def __len__(self):
        return self.bits


def count(hashes):
    """http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694"""
    return len(hashes) / np.sum(hashes / MAX) - 1.0


def jaccard(mh1, mh2):
    return np.sum(mh1 == mh2) / len(mh1)


if __name__ == "__main__":
    mh = MinHash(32)
    set1 = set("Th is is th es et".split())
    set2 = set("Th is is th es at az dr go he".split())
    a = mh(set1)
    b = mh(set2)
    print(jaccard(a, b))
    print(len(set1 & set2) / len(set1 | set2))
    print(len(set1))
    print(count(a))
    print(len(set2))
    print(count(b))
