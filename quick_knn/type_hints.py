from typing import Union, Iterable, NewType, Any, Callable
import numpy as np

Signature = NewType('Signature', np.ndarray)
Vector = NewType('Vector', np.ndarray)
Key = NewType('Key', Any)
Hashable = Union[str, bytes, Iterable[Union[str, bytes]]]
Integrable = Callable[[float], float]
