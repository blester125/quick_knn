import numpy as np
from quick_knn.type_hints import Signature

class RandomHyperplanes(object):

    def __init__(self, bits: int, dim: int):
        super().__init__()
        self.bits = bits
        self.dim = dim
        self.planes = np.random.randn(self.dim, self.bits)

    def __call__(self, data: np.ndarray) -> Signature:
        return self.signature(data)

    def signature(self, data: np.ndarray) -> Signature:
        return (np.dot(data, self.planes) >= 0).astype(np.uint8)


def cosine(query: Signature, dataset: Signature) -> float:
    return 1 - np.mean(np.logical_xor(query, dataset), axis=1)


if __name__ == "__main__":
    dim = 20
    bits = 1024
    rh = RandomHyperplanes(bits, dim)

    pt1 = np.random.randn(dim)
    pt2 = pt1 * 5

    pts = np.vstack([pt1, pt2])

    sigs = rh(pts)

    np.testing.assert_allclose(sigs[0, :], sigs[1, :])
    print("Asserts Passed!")
