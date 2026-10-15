"""The NumPy-compatible random API used by environments and deterministic tests."""
from typing import Any, Protocol
import numpy as np


class RandomSource(Protocol):
    @property
    def bit_generator(self) -> np.random.BitGenerator: ...
    def random(self, *args: Any, **kwargs: Any) -> Any: ...
    def integers(self, *args: Any, **kwargs: Any) -> Any: ...
    def choice(self, *args: Any, **kwargs: Any) -> Any: ...
    def shuffle(self, *args: Any, **kwargs: Any) -> Any: ...
    def permutation(self, *args: Any, **kwargs: Any) -> Any: ...
    def uniform(self, *args: Any, **kwargs: Any) -> Any: ...


class RandomProxy:
    """Expose the RNG API explicitly for wrappers forwarding with __getattr__."""
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    @property
    def bit_generator(self) -> np.random.BitGenerator:
        return self.__getattr__("bit_generator")

    def random(self, *args: Any, **kwargs: Any) -> Any:
        return self.__getattr__("random")(*args, **kwargs)

    def integers(self, *args: Any, **kwargs: Any) -> Any:
        return self.__getattr__("integers")(*args, **kwargs)

    def choice(self, *args: Any, **kwargs: Any) -> Any:
        return self.__getattr__("choice")(*args, **kwargs)

    def shuffle(self, *args: Any, **kwargs: Any) -> Any:
        return self.__getattr__("shuffle")(*args, **kwargs)

    def permutation(self, *args: Any, **kwargs: Any) -> Any:
        return self.__getattr__("permutation")(*args, **kwargs)

    def uniform(self, *args: Any, **kwargs: Any) -> Any:
        return self.__getattr__("uniform")(*args, **kwargs)
