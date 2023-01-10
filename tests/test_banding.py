"""Tests for banding calculations."""

import numpy as np
from quick_knn import banding


def test_integrate_x_squared():
    def func(x):
        return x**2

    gold = 2 / 3
    area = banding.integrate(func, -1, 1)
    np.testing.assert_allclose(area, gold, rtol=1e-6)


def test_integrate_sin_x():
    def func(x):
        return np.sin(x)

    gold = 0.0
    area = banding.integrate(func, -1, 1)
    np.testing.assert_allclose(area, gold, atol=1e-6)


def test_integrate_vectorized():
    intercept = np.array([2, 3, 4])

    def func(x):
        return x**3 + intercept

    gold = np.array([4, 6, 8])
    area = banding.integrate(func, -1, 1)
    np.testing.assert_allclose(area, gold)


def test_and_or():
    r = 4
    b = 4

    ps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = [0.0064, 0.0320, 0.0985, 0.2275, 0.4260, 0.6666, 0.8785, 0.9860]

    for p, res in zip(ps, results):
        s = banding.AndOrSensitivity(d1=p, d2=1 - p, p1="1 - d1", p2="1 - d2", r=r, b=b)
        s2 = s.evaluate()
        np.testing.assert_allclose(round(s2.p2, 4), res)
