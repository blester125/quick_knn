"""Hash Families for Cosine Distance.

http://snap.stanford.edu/class/cs246-2015/slides/04-lsh_theory.pdf
"""


import numpy as np
from quick_knn.hashes import HashFamily
from quick_knn.utils import Sensitivity


class RandomHyperplanesBase(HashFamily):
    def __init__(self, signature_size: int, seed: int):
        super().__init__(signature_size, seed)

    @property
    def name(self):
        return "cosine"

    @property
    def sensitivity(self) -> Sensitivity:
        return Sensitivity(
            d1="d1",
            d2="d2",
            p1="1 - (d1 / 180)",
            p2="1 - (d2 / 180)",
            p1_func=lambda d1: 1 - (d1 / 180),
            p2_func=lambda d2: 1 - (d2 / 180),
        )

    def similarity(self, query, data) -> float:
        # Hamming distance between signatures.
        return 1 - np.mean(np.logical_xor(query, data), axis=-1)


class RandomHyperplanes(RandomHyperplanesBase):
    """Cosine Distance via Random Hyperplane hashes."""

    def __init__(self, signature_size: int, feature_size: int, seed: int):
        super().__init__(signature_size, seed)
        self.feature_size = feature_size
        rng = np.random.RandomState(seed)
        self.hyper_planes = rng.normal(size=(feature_size, signature_size))

    def hash(self, x):
        return ((x @ self.hyper_planes) >= 0).astype(np.uint8)

    def __repr__(self):
        return f"{self.__class__.__name__}(distance={self.name}, signature_size={self.signature_size}, feature_size={self.feature_size}, seed={self.seed}, sensitivity={self.sensitivity})"


if __name__ == "__main__":
    rhp = RandomHyperplanes(32, 512, 0)
    print(rhp.hyper_planes.shape)
    print(rhp)
