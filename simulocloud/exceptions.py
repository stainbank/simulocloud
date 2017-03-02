"""exceptions.py
Custom exceptions for simulocloud
"""

class SimulocloudException(Exception):
    """Base exception for simulocloud."""
    pass


class PointcloudException(SimulocloudException):
    """Base exception for the pointcloud module."""
    pass


class VisualiseException(SimulocloudException):
    """Base exception for the visualise module."""
    pass

class EmptyPointCloud(PointcloudException, ValueError):
    """An empty PointCloud would be created against user wishes."""
    pass

class AxesError(VisualiseException):
    """Base exception for errors relating to the axes argument."""
    pass

class InvalidAxesDims(AxesError, ValueError):
    """When axes argument is not two or three dimensions."""
    pass

class BadAxes(AxesError, TypeError):
    """When axes argument is not a string"""
    pass
