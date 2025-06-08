from abc import ABC, abstractmethod


class Shape(ABC):
    """Class for feeding shapes"""

    @abstractmethod
    def contains(self, data_points, fish_key, day=None):
        """Checks which points are inside the shape"""
        pass

