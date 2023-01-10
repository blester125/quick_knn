"""Tools for banding choices."""

import dataclasses
import functools
from typing import Optional, Union
import numpy as np
from quick_knn.utils import Sensitivity


def integrate(func, start: float, end: float, dt: float = 0.001) -> float:
    """Midpoint Riemann Sum Integration."""
    points = np.reshape(np.arange(start, end, dt), (-1, 1)) + 0.5 * dt  # [dts, 1]
    # Areas of the squares, func(points) is height at midpoint and dt is width.
    # Note: func is vectorized to solve for all points at once. Additionally, it
    # is generally partial'ed to solve for multiple settings at once.
    area = func(points) * dt  # [dts, rs]
    return np.sum(area, axis=0)  # [rs]


def false_positive_probability(p: float, r: int, b: int) -> float:
    # p is the probability of the hashes agreeing.
    # p^r  is the probability of matching all r times (in the band).
    # (1 - s^r) is the probability of any disagreement (in the band).
    # (1 - s^r)^b is the probability of disagreement in all bands.
    # 1 - (1 - s^4)^b is the probability of matching in at least one band.
    return 1 - (1 - p**r) ** b


def false_negative_probability(p: float, r: int, b: int) -> float:
    # As above, (1 - s^r)^b is the probability of mismatch in all bands.
    return (1 - p**r) ** b


def optimize_r_and_b(
    probability_threshold: float, signature_size: int, false_positive_weight: float
):
    false_negative_weight = 1 - false_positive_weight

    # Enumerate all possible b and r values based on signature size.
    # Note: It is possible for b * r < signature_size if that gives better
    # probability bounds.
    bs = []
    rs = []
    for b in range(1, signature_size + 1):
        max_r = signature_size // b
        for r in range(1, max_r + 1):
            bs.append(b)
            rs.append(r)
    bs = np.array(bs)
    rs = np.array(rs)

    # Integrate from 0 to d1 to see how probable false positives are.
    false_positive_func = functools.partial(false_positive_probability, r=rs, b=bs)
    false_positive = integrate(false_positive_func, 0.0, probability_threshold)

    # Integrate from d2 to 1 to see how probable false negatives are.
    false_negative_func = functools.partial(false_negative_probability, r=rs, b=bs)
    false_negative = integrate(false_negative_func, probability_threshold, 1.0)

    # Mix errors based on user weights if which they care about more.
    errors = (
        false_positive_weight * false_positive + false_negative_weight * false_negative
    )

    # Grab the settings that minimize the error.
    min_idx = np.argmin(errors)
    return rs[min_idx], bs[min_idx]


@dataclasses.dataclass
class AndOrSensitivity(Sensitivity):
    r: Union[str, int] = "r"
    b: Union[str, int] = "b"

    def __post_init__(self):
        # If we are storing the symbolic values add the r and b transform.
        if isinstance(self.p1, str):
            self.p1 = f"1-(1-({self.p1})**r)**b"
        if isinstance(self.p2, str):
            self.p2 = f"1-(1-({self.p2})**r)**b"

    def __str__(self):
        return f"{self.__class__.__name__}(d1={self.d1}, d2={self.d2}, p1={self.p1}, p2={self.p2}, r={self.r}, b={self.b})"

    def evaluate(
        self,
        d1: Optional[float] = None,
        d2: Optional[float] = None,
        r: Optional[int] = None,
        b: Optional[int] = None,
    ):
        if d1 is None:
            if isinstance(self.d1, str):
                raise ValueError(
                    "d1 must be provided if not set on the sensitivity object."
                )
            d1 = float(self.d1)
        if d2 is None:
            if isinstance(self.d2, str):
                raise ValueError(
                    "d1 must be provided if not set on the sensitivity object."
                )
            d2 = float(self.d2)
        # Allow you to pre-set r and b and use them in evaluations.
        if r is None:
            if isinstance(self.r, str):
                raise ValueError(
                    "r must be provided if not set on the sensitivity object."
                )
            r = int(self.r)
        if b is None:
            if isinstance(self.b, str):
                raise ValueError(
                    "b must be provided if not set on the sensitivity object."
                )
            b = int(self.b)
        return self.__class__(
            d1=d1,
            d2=d2,
            p1=eval(self.p1, {}, {"d1": d1, "r": r, "b": b}),
            p2=eval(self.p2, {}, {"d2": d2, "r": r, "b": b}),
            p1_func=self.p1_func,
            p2_func=self.p2_func,
            r=r,
            b=b,
        )


if __name__ == "__main__":
    s = AndOrSensitivity(
        0.2, 0.8, "1 - d1", "1 - d2", lambda d1: 1 - d1, lambda d2: 1 - d2, r=4, b=4
    )
    print(s)
    print(s.evaluate())
    r, b = optimize_r_and_b(0.2, 16, 0.9)
    d = s.evaluate(r=r, b=b)
    print(d)
    print(s.evaluate(r=2, b=6))
