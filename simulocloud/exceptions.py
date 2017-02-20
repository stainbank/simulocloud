"""exceptions.py
Custom exceptions for simulocloud
"""

class SimulocloudException(Exception):
    """Base exception for simulocloud."""
    pass


class PointcloudException(SimulocloudException):
    """Base exception for the pointcloud module."""
    pass


class EmptyPointCloud(PointcloudException, ValueError):
    """An empty PointCloud would be created against user wishes."""
    pass
