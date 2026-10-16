"""Trigger applied to a collated batch. Returns a new batch and does not modify the input."""

from abc import ABC, abstractmethod

from torch import Tensor


class Trigger(ABC):
    @abstractmethod
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return a triggered copy of the batch."""
