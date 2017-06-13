"""exceptions.py
Custom exceptions for simulocloud
"""

class SimulocloudException(Exception):
    """Base exception for simulocloud."""
    pass


class PointCloudException(SimulocloudException):
    """Base exception for the `simulocloud.pointcloud.Pointcloud` class."""
    pass

class TileException(PointCloudException):
    """Base exceptions for the `simulocloud.tiles.Tile` subclass."""
    pass

class TilesGridException(SimulocloudException):
    """Base exceptions for the `simulocloud.tiles.TilesGrid` class."""
    pass

class VisualiseException(SimulocloudException):
    """Base exception for the visualise module."""
    pass

class EmptyPointCloud(PointCloudException, ValueError):
    """An empty PointCloud is being created or summarised."""
    pass

class DimsError(VisualiseException):
    """Base exception for errors relating to the dims argument."""
    pass

class WrongNDims(DimsError, ValueError):
    """When dims argument does not present two or three dimensions."""
    pass

class BadDims(DimsError, TypeError):
    """When dims argument is not a string"""
    pass
