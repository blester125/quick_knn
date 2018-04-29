import numpy as np


class RandomHyperplanes(object):

    def __init__(self, bits, dim):
        super().__init__()
        self.bits = bits
        self.dim = dim
        self.planes = np.random.randn(self.dim, self.bits)

    def __call__(self, data):
        return self.signature(data)

    def signature(self, data):
        return (np.dot(data, self.planes) >= 0).astype(np.uint8)


def cosine(query, dataset):
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
