"""Hash Families to build LSH's from.

Note: Each HashFamily actually produces `signature_size` independent hashes.
"""

import abc
from quick_knn.utils import Sensitivity
from quick_knn.types import Signature


class HashFamily(metaclass=abc.ABCMeta):
    def __init__(self, signature_size: int, seed: int):
        self._signature_size = signature_size
        self.seed = seed

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the distance function this hash family approximates."""

    @property
    def signature_size(self) -> int:
        """The size of the signatures this hash produces."""
        return self._signature_size

    @property
    @abc.abstractmethod
    def sensitivity(self) -> Sensitivity:
        """Information about the sensitivity of the hash family."""

    @abc.abstractmethod
    def hash(self, x) -> Signature:
        """Convert `x` to its signature."""

    @abc.abstractmethod
    def similarity(self, query: Signature, data: Signature) -> float:
        """Calculate the approximate distance between two signatures."""
