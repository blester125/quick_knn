import numpy as np
from quick_knn.lsh import integrate

def test_integrate_x_squared():
    def func(x):
        return x ** 2
    gold = 2 / 3
    area = integrate(func, -1, 1)
    np.testing.assert_allclose(area, gold, rtol=1e-6)

def test_integrate_sin_x():
    def func(x):
        return np.sin(x)
    gold = 0.0
    area = integrate(func, -1, 1)
    np.testing.assert_allclose(area, gold, atol=1e-6)

def test_integrate_vectorized():
    intercept = np.array([2, 3, 4])
    def func(x):
        return x ** 3 + intercept
    gold = np.array([4, 6, 8])
    area = integrate(func, -1, 1)
    np.testing.assert_allclose(area, gold)
