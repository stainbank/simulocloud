"""
pointcloud

Read in and store point clouds.
"""

import numpy as np
from laspy.file import File
from laspy.header import Header, VLR
from collections import namedtuple
from .exceptions import EmptyPointCloud

_HEADER_DEFAULT = {'data_format_id': 3,
                   'x_scale': 2.5e-4,
                   'y_scale': 2.5e-4,
                   'z_scale': 2.5e-4,
                   'software_id': "simulocloud"[:32].ljust(32, '\x00'),
                   'system_id': "CREATION"[:32].ljust(32, '\x00')}

_VLR_DEFAULT = {'user_id': 'LASF_Projection\x00',
               'record_id': 34735,
               'VLR_body': ('\x01\x00\x01\x00\x00\x00\x03\x00\x01\x04\x00'
                            '\x00\x01\x00\x02\x00\x00\x04\x00\x00\x01\x00'
                            '\x03\x00\x00\x08\x00\x00\x01\x00\xe6\x10'),
               'description': 'GeoKeyDirectoryTag (mandatory)\x00\x00',
               'reserved': 43707}

class PointCloud(object):
    """ Contains point cloud data """
    
    dtype = np.float64

    def __init__(self, xyz, header=None):
        """Store 3D point coordinates in a structured array.
        
        Arguments
        ---------
        xyz: sequence of len 3
            equal sized sequences specifying 3D point coordinates (xs, ys, zs)
        header: laspy.header.Header instance
            base header to use for output
        
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
        
        if header is not None:
            self._header = header

    def __len__(self):
        """Number of points in point cloud"""
        return len(self.points)
    
    """ Constructor methods """
 
    @classmethod
    def from_las(cls, fpath):
        """Initialise PointCloud from .las file.
    
        Arguments
        ---------
        fpath: str
            filepath of .las file containing 3D point coordinates
       
        """
        with File(fpath) as f:
            return cls.from_laspy_File(f)

    @classmethod
    def from_laspy_File(cls, f):
        """Initialise PointCloud from a laspy File.
        
        Arguments
        ---------
        f: `laspy.file.File` instance
            file object must be open, and will remain so
        
        """
        return PointCloud((f.x, f.y, f.z), header=f.header.copy())
    
    """ Instance methods """
    @property
    def x(self):
        """The x dimension of point coordinates."""
        return self.points['x']

    @property
    def y(self):
        """The y dimension of point coordinates."""
        return self.points['y']

    @property
    def z(self):
        """The z dimension of point coordinates."""
        return self.points['z']

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

    @property
    def header(self):
        """Create a valid header describing pointcloud for output to .las.

        Returns
        -------
        header: laspy.header.Header instance
            header generated from up-to-date point cloud information 
        
        """
        header = _HEADER_DEFAULT.copy()
        bounds = self.bounds
        header.update({'point_return_count': [len(self), 0, 0, 0, 0],
                       'x_offset': round(bounds.minx),
                       'y_offset': round(bounds.miny),
                       'z_offset': round(bounds.minz),
                       'x_min': bounds.minx,
                       'y_min': bounds.miny,
                       'z_min': bounds.minz,
                       'x_max': bounds.maxx,
                       'y_max': bounds.maxy,
                       'z_max': bounds.maxz})

        return Header(**header)
    
    def crop(self, bounds, return_empty=False):
        """Crop point cloud to (lower-inclusive, upper-exclusive) bounds.
        
        Arguments
        ---------
        bounds: `Bounds` namedtuple
            (minx, miny, minz, maxx, maxy, maxz) to crop within
        return_empty: bool (defaul: False)
            whether to allow empty pointclouds to be created or raise an
            EmptyPointCloud exception        
        
        Returns
        -------
        PointCloud instance
            new object containing only points within specified bounds
        
        """
        # Build results using generator to limit memory usage
        out_of_bounds = np.zeros(len(self))
        for comparison in iter_out_of_bounds(self.points, bounds):
            out_of_bounds = np.logical_or(comparison, out_of_bounds)
        
        # Deal with empty pointclouds
        if out_of_bounds.all():
            if return_empty:
                return PointCloud([[], [], []])
            else:
                raise EmptyPointCloud, "No points in crop bounds:\n{}".format(
                                            bounds)
         
        return PointCloud(self.points[~out_of_bounds])

    def to_txt(self, fpath):
        """Export point cloud coordinates as 3-column (xyz) ASCII file.
    
        Arguments
        ---------
        fpath: str
            path to file to write 
        
        """
        np.savetxt(fpath, self.arr.T)

    def to_las(self, fpath):
        """Export point cloud coordinates to .las file.

        Arguments
        ---------
        fpath: str
            path to file to write
        
        """
        with File(fpath, mode='w', header=self.header,
                  vlrs=[VLR(**_VLR_DEFAULT)]) as f:
            f.x = self.points['x']
            f.y = self.points['y']
            f.z = self.points['z']

    def downsample(self, n):
        """Randomly sample the point cloud.
        
        Arguments
        ---------
        n: int
            number of points in sample
        
        Returns
        -------
        PointCloud
            of len n (or len of this pointcloud if it is <=n)
        
        """
        n = min(n, len(self))
        return PointCloud(np.random.choice(self.points, n, replace=False))

# Container for bounds box surrounding PointCloud
Bounds = namedtuple('Bounds', ['minx', 'miny', 'minz', 'maxx', 'maxy', 'maxz'])

def iter_out_of_bounds(points, bounds):
    """Iteratively determine point coordinates outside of bounds.

    Arguments
    ---------
    points: numpy.ndarray
        structured array containing 'x', 'y' and 'z' point coordinates
    bounds: `Bounds` namedtuple
        (minx, miny, minz, maxx, maxy, maxz) to test point coordinates against
    
    Returns
    -------
    generator (len 6)
        each iteration yields a boolean numpy.ndarray describing whether,
        seperately, each dimension of (x,y,z) point coordinates are outside
        each of the respective lower and upper bound values
        sequence order is identical to bounds
    
    Notes
    -----
    Comparisons are python-like, i.e.:
        x < minx
        x >= maxx
        All coordinate values compare False to `None`.
    """
    # Set up comparison elements in same order as bounds
    comparison_funcs = (np.less,)*3 + (np.greater_equal,)*3
    coords = ('x', 'y', 'z')*2
    
    for compare, c, bound in zip(comparison_funcs, coords, bounds):
        if bound is None: # None is a permissive bound
            yield np.zeros_like(points[c], dtype=bool)
        else:
            yield compare(points[c], bound)
