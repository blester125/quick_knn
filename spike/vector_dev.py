import numpy as np

def signature_bit(data, planes):
    sig = 0
    for p in planes:
        sig <<= 1
        if np.dot(data, p) >= 0:
            sig |= 1
    return sig

def v_sig_bit(data, planes):
    return (np.dot(planes, data) >= 0).astype(np.uint8)

def bitcount(n):
    count = 0
    while n:
        count += 1
        n = n & (n - 1)
    return count

def vbitcount(query, vects):
    return np.mean(np.logical_xor(vects, query), axis=0)

def random_hyperplanes(dims, bits):
    planes = np.random.randn(bits, dims)

    def sketch(data):
        return (np.dot(planes, data) >= 0).astype(np.unit8)

    return sketch

def cosine_dist(dataset, sketch):
    vects = sketch(dataset)

    def dist(query):
        return np.mean(np.logical_xor(vects, query), axis=0)

    return dist

def length(v):
    return np.sqrt(np.dot(v, v))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", "-d", type=int, default=20)
    parser.add_argument("--bits", "-b", type=int, default=1204)
    args = parser.parse_args()

    pt1 = np.random.randn(args.dim)
    pt2 = np.random.randn(args.dim)
    pt3 = np.random.randn(args.dim)

    pts = np.hstack([pt1.reshape(-1, 1), pt2.reshape(-1, 1), pt3.reshape(-1, 1)])

    planes = np.random.randn(args.bits, args.dim)

    sig1 = signature_bit(pt1, planes)
    sig2 = signature_bit(pt2, planes)
    sig3 = signature_bit(pt3, planes)
    sigs = v_sig_bit(pts, planes)

    cosine_hash_12 = 1 - bitcount(sig1 ^ sig2) / args.bits
    cosine_hash_23 = 1 - bitcount(sig1 ^ sig3) / args.bits

    vector_hash = 1 - vbitcount(sigs[:, 0, np.newaxis], sigs[:, 1:])

    np.testing.assert_allclose(cosine_hash_12, vector_hash[0])
    np.testing.assert_allclose(cosine_hash_23, vector_hash[1])

    print("Asserts Passed!")
