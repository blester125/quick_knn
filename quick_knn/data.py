from typing import Union, Iterable, NewType, Any
import numpy as np

signature = NewType('signature', np.ndarray)
Key = NewType('Key', Any)
hashable = Union[str, bytes, Iterable[Union[str, bytes]]]
