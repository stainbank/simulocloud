"""
pointcloud

Read in and store point clouds.
"""

import numpy as np
import laspy
from collections import namedtuple

class PointCloud(object):
    """ Contains point cloud data """

    dtype = np.float64

    def __init__(self, xyz):
        """Store 3D point coordinates in a structured array.
        
        Arguments
        ---------
        xyz: sequence of len 3
            equal sized sequences specifying 3D point coordinates (xs, ys, zs)
        
        Example
        -------
        >>> from numpy.random import rand
        >>> n = 5 # number of points
        >>> x, y, z = rand(n)*100, rand(n)*100., rand(n)*20
        >>> pc = PointCloud((x,y,z))
        >>> pc.points
        array([( 75.37742432,  20.33372458,  14.73631503),
               ( 39.34924712,  29.56923584,  12.15410051),
               ( 37.11597209,  47.5210436 ,   7.08069784),
               ( 46.30703381,  62.75060038,  17.70324372),
               ( 25.92662908,  55.45793312,   5.95560623)], 
              dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
        
        Notes
        -----
        The use of the constructor methods (`PointCloud.from...`) is preferred.
        """
        # Combine x, y and z into (flat) structured array 
        self.points = np.column_stack(xyz).ravel().view(
            dtype=[('x', self.dtype), ('y', self.dtype), ('z', self.dtype)])
    
    """ Constructor methods """
    
    @classmethod
    def from_las(cls, fpath):
        """Initialise PointCloud from .las file.

        Arguments
        ---------
        fpath: str
            filepath of .las file containing 3D point coordinates
       
        """
        with laspy.file.File(fpath) as f:
            return cls.from_laspy_File(f)

    @classmethod
    def from_laspy_File(cls, f):
        """Initialise PointCloud from a laspy File.
        
        Arguments
        ---------
        f: `laspy.file.File` instance
            file object must be open, and will remain so
        
        """
        return PointCloud((f.x, f.y, f.z))
    
    """ Instance methods """
    
    @property
    def arr(self):
        """Get point coordinates as unstructured n*3 array).
        
        Returns
        -------
        np.ndarray with shape (npoints, 3)

        """
        return self.points.view(self.dtype).reshape(-1, 3)

    @property
    def bounds(self):
        """Boundary box surrounding PointCloud.
        
        Returns
        -------
        namedtuple (minx, miny, minz, maxx, maxy, maxz)
        """
        p = self.points
        return Bounds(np.min(p['x']), np.min(p['y']), np.min(p['z']),
                      np.max(p['x']), np.max(p['y']), np.max(p['z']))


# Container for bounds box surrounding PointCloud
Bounds = namedtuple('Bounds', ['minx', 'miny', 'minz', 'maxx', 'maxy', 'maxz'])
