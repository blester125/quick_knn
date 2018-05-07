import os
import time
import pickle
import random
import argparse
import numpy as np
from quick_knn import LSH, RandomHyperplanes

GLOVE_LOC = "data/glove.6B.300d.txt"

def read_glove(file_name):
    vocab = []
    vectors = []
    with open(file_name) as f:
        for line in f:
            line = line.rstrip("\n")
            word, *vec = line.split(" ")
            if len(vec) != 300:
                continue
            vocab.append(word)
            vectors.append(np.array(list((map(float, vec)))))
    return vocab, np.array(vectors)


def shuffle(vocab, vectors):
    together = list(zip(vocab, vectors))
    random.shuffle(together)
    return list(zip(*together))

def get_glove(vocab_cache="data/cache.p", vector_cache="data/cache.npy"):
    if os.path.exists(vocab_cache) and os.path.exists(vocab_cache):
        vocab = pickle.load(open(vocab_cache, "rb"))
        vectors = np.load(vector_cache)
    else:
        vocab, vectors = read_glove(GLOVE_LOC)
        d = (np.sum(vectors ** 2, axis=1) ** (0.5))
        vectors = (vectors.T / d).T
        pickle.dump(vocab, open(vocab_cache, "wb"))
        np.save(vector_cache, vectors)
    return vocab, [v for v in vectors]

def f1(gold, res):
    tp = len([x for x in gold if x in res])
    p = tp / (len(res) + 1e-9)
    r = tp / (len(gold) + 1e-9)
    f = (2 * p * r) / (p + r + 1e-9)
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"F1: {f}")
    return f

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh", "-t", type=float, default=.5)
    parser.add_argument("--bits", "-b", type=int, default=64)
    parser.add_argument("--seed", "-s", type=int, default=1337)
    args = parser.parse_args()
    random.seed(args.seed)

    vocab, vectors = get_glove()
    vocab, vectors = shuffle(vocab, vectors)
    vectors = np.vstack(vectors)

    query_vocab, data_vocab = vocab[:5], vocab[5:]
    query_vectors, data_vectors = vectors[:5, :], vectors[5:, :]

    # Gold
    t0 = time.time()
    dist = np.dot(query_vectors, data_vectors.T)
    gold = []
    for d in dist:
        gold.append(np.where(d > args.thresh)[0])
    print(f"Naive way: {time.time() - t0}")

    t0 = time.time()
    lsh = LSH(args.thresh, args.bits)
    rh = RandomHyperplanes(args.bits, data_vectors.shape[1])
    sigs = rh(data_vectors)
    for i, sig in enumerate(sigs):
        lsh.insert(i, sig)
    build_time = time.time() - t0

    t0 = time.time()
    qs = rh(query_vectors)
    res = []
    for q in qs:
        res.append(lsh.query(q))
    query_time = time.time() - t0

    for i, (g, r) in enumerate(zip(gold, res)):
        print(f"Query: {i + 1}")
        f1(g, r)
        print()

    print(f"LSH Build time: {build_time} Query time: {query_time} Total time {build_time + query_time}")

if __name__ == "__main__":
    main()
