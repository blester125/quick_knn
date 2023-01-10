#!/usr/bin/env python3

import pickle
import tempfile
import numpy as np
from quick_knn.hashes import pooling_random_hyperplanes


def test_saving_and_load():
    sig_size = 32
    pool_size = 10_000
    seed = 0
    prhp1 = pooling_random_hyperplanes.PoolingRandomHyperPlanes(
        sig_size, pool_size, seed
    )

    with tempfile.NamedTemporaryFile(mode="w+b") as wf:
        pickle.dump(prhp1, wf)
        wf.seek(0)
        prhp2 = pickle.load(wf)

    np.testing.assert_allclose(prhp1.pool, prhp2.pool)
    np.testing.assert_allclose(prhp1.hash_offsets, prhp2.hash_offsets)
