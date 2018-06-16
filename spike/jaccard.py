import re
import time
import random
import argparse
from functools import partial
from quick_knn import LSH, MinHash

CRAN_FILE = "data/cran.all.1400"

def read_cran(file_name):
    regex = re.compile(r"\.W(.*?)\.I", re.MULTILINE | re.DOTALL)
    dataset = open(file_name).read()
    data = []
    for m in regex.finditer(dataset):
        data.append(m.groups(1)[0])
    data = filter(lambda x: x != "\n", data)
    data = map(lambda x: " ".join(x.replace("\n", " ").split()), data)
    return data

def shingle(data):
    n = 2
    s = set()
    for i in range(n, len(data) + 1):
        s.add(data[i - n:i].lower().encode("utf-8"))
    return s

def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)

def _find_best(query, thresh= 0.51, dataset=None):
    results = [(i, jaccard(query, data)) for i, data in enumerate(dataset)]
    results = filter(lambda x: x[1] >= thresh, results)
    return list(results)

def f1(gold, res):
    tp = len([x for x in gold if x[0] in res])
    p = tp / (len(res) + 1e-9)
    r = tp / (len(gold) + 1e-9)
    f = (2 * p * r) / (p + r + 1e-9)
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"F1: {f}")
    return f

def encode(d, mh):
    # This is a bad function but cleans up our code
    my = None
    for x in d:
        my = mh.signature(x, my)
    return my


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh", "-t", default = 0.59, type=float)
    parser.add_argument("--bits", "-b", default = 1024, type=int)
    parser.add_argument("--seed", "-s", default=5777, type=int)
    parser.add_argument("--fp_weight", "-fp", default=0.5, type=float)
    args = parser.parse_args()
    random.seed(args.seed)

    data = list(read_cran(CRAN_FILE))
    random.shuffle(data)
    queries, data = data[:5], data[5:]
    queries = list(map(shingle, queries))
    data = list(map(shingle, data))

    # Gold
    find_best = partial(_find_best, thresh=args.thresh, dataset=data)
    t0 = time.time()
    gold = list(map(find_best, queries))
    print(f"Naive Time: {time.time() - t0}")

    # LSH
    lsh = LSH(args.thresh, args.bits, args.fp_weight)
    print(lsh)
    mh = MinHash(args.bits)

    t0 = time.time()
    for i, d in enumerate(data):
        my = encode(d, mh)
        lsh.insert(i, my)
    build_time = time.time() - t0

    t0 = time.time()
    res = []
    for query in queries:
        my = encode(query, mh)
        res.append(lsh.query(my))
    query_time = time.time() - t0

    m_score = []
    for i, (g, r) in enumerate(zip(gold, res)):
        print(f"Query: {i + 1}")
        print("Mine")
        m_score.append(f1(g, r))
        print()

    print(f"\nLSH Build time: {build_time}, Query time: {query_time} Total time {build_time + query_time}")

if __name__ == "__main__":
    main()
