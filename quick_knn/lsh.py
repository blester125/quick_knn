from collections import defaultdict
import numpy as np


def integrate(func, a, b, dt=0.001):
    area = 0.0
    while a < b:
        area += func(a + 0.5 * dt) * dt
        a += dt
    return area


def fp_prob(thresh, b, r):
    def probs(s):
        return 1 - (1 - s ** r) ** b

    return integrate(probs, 0.0, thresh)


def fn_prob(thresh, b, r):
    def probs(s):
        return 1 - (1 - (1 - s ** r) ** b)
    return integrate(probs, thresh, 1.0)


def opt_b_r(thresh, bits, fp_weight, fn_weight):
    bs = []
    rs = []
    for b in range(bits):
        max_r = bits // (b + 1)
        for r in range(max_r):
            bs.append(b + 1)
            rs.append(r + 1)
    bs = np.array(bs)
    rs = np.array(rs)
    fp = fp_prob(thresh, bs, rs)
    fn = fn_prob(thresh, bs, rs)
    error = fp * fp_weight + fn * fn_weight
    idx = np.argmin(error)
    return bs[idx], rs[idx]


class LSH(object):

    def __init__(self, threshold=0.51, bits=32):
        super().__init__()

        self.threshold = threshold
        self.bits = bits
        self.b, self.r = opt_b_r(threshold, bits, 0.5, 0.5)

        self.ranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.tables = [defaultdict(set) for _ in range(self.b)]

    def insert(self, key, sig):
        parts = [LSH.hashable(sig[start:end]) for start, end in self.ranges]
        for part, table in zip(parts, self.tables):
            table[part].add(key)

    def query(self, sig):
        cands = set()
        for (start, end), table in zip(self.ranges, self.tables):
            part = LSH.hashable(sig[start:end])
            for key in table[part]:
                cands.add(key)
        return list(cands)

    @staticmethod
    def hashable(hs):
        return bytes(hs.data)
