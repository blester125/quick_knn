#!/usr/bin/env python3

import dataclasses
from typing import Optional, Union, Callable


@dataclasses.dataclass
class Sensitivity:
    d1: Union[str, float]
    d2: Union[str, float]
    p1: Union[str, float]
    p2: Union[str, float]
    p1_func: Callable[[float], float] = None
    p2_func: Callable[[float], float] = None

    def evaluate(self, d1: Optional[float] = None, d2: Optional[float] = None):
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
        return self.__class__(
            d1=d1,
            d2=d2,
            p1=self.p1_func(d1) if self.p1_func else eval(self.p1, {}, {"d1": d1}),
            p2=self.p2_func(d2) if self.p2_func else eval(self.p2, {}, {"d2": d2}),
        )

    def __str__(self):
        return f"{self.__class__.__name__}(d1={self.d1}, d2={self.d2}, p1={self.p1}, p2={self.p2})"


@dataclasses.dataclass
class SensitivityExplanitionClass:
    d1: str = "distance(x, y) ≤ d1"
    d2: str = "distance(x, y) ≥ d2"
    p1: str = "P[h(x) = h(y)] ≥ p1"
    p2: str = "P[h(x) = h(y)] ≤ p2"

    def __str__(self):
        return f"LSH (d1, d2, p1, p2) Sensitivity: If {self.d1} then {self.p1}, If {self.d2} then {self.p2}."


SensitivityExplanition = SensitivityExplanitionClass()


if __name__ == "__main__":
    print(SensitivityExplanition)
    s = Sensitivity("d1", "d2", "1 - (d1 / 180)", "1 - (d2 / 180)")
    print(s)
    print(s.evaluate(0.0001, 0.2))
