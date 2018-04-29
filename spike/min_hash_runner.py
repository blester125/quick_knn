import numpy as np

from quick_knn import LSH, MinHash

set1 = set(map(lambda x: x.encode(), ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for', 'estimating', 'the', 'similarity', 'between', 'datasets']))
set2 = set(map(lambda x: x.encode(), ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for', 'estimating', 'the', 'similarity', 'between', 'documents']))
set3 = set(map(lambda x: x.encode(), ['minhash', 'is', 'probability', 'data', 'structure', 'for', 'estimating', 'the', 'similarity', 'between', 'documents']))
set4 = set(map(lambda x: x.encode(), ["totally", "different", "one"]))

sets = [set2, set3, set4]
query = set1

lsh = LSH(0.51, 32)
mh = MinHash(32)

for i, s in enumerate(sets):
    my = None
    for x in s:
        my = mh.signature(x, my)
    lsh.insert(str(i), my)


my = None
for x in query:
    my = mh.signature(x, my)


c = lsh.query(my)

print(c)
