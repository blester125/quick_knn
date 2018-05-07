import numpy as np

from quick_knn import RandomHyperplanes, LSH

dims = 300
bits = 32
len_vects = 100

vects = np.random.randn(len_vects, dims)
query = np.random.randn(1, dims)

rh = RandomHyperplanes(bits, dims)

sigs = rh(vects)

lsh = LSH(0.51, bits)

def cosine(v1, v2):
    a = np.dot(v1, v2) / np.sqrt(np.dot(v1,v1)) / np.sqrt(np.dot(v2,v2))
    return 1 - np.arccos(a) / np.pi

real = []
for i, (sig, vec) in enumerate(zip(sigs, vects)):
    lsh.insert(str(i), sig)
    cos = cosine(vec, np.squeeze(query))
    if cos > .51:
        real.append(str(i))

print(len(real))

results = set(lsh.query(np.squeeze(rh(query))))
print(len(results))

assert len(results) < vects.shape[0]

print("Asserts Passed!")
