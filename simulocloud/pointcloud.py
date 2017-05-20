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

_DTYPE = np.float64

class PointCloud(object):
    """ Contains point cloud data """
    
    dtype = _DTYPE

    def __init__(self, xyz, header=None):
        """Create PointCloud with 3D point coordinates stored in a (3*n) array.
        
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
        # Coerce None to empty array
        if xyz is None:
            xyz = [[], [], []]
        
        # Store points as 3*n array
        x, y, z = xyz # ensure only 3 coordinates
        self.arr = np.stack([x, y, z])

        if header is not None:
            self._header = header

    def __len__(self):
        """Number of points in point cloud"""
        return self.arr.shape[1]

    def __add__(self, other):
        """Concatenate two PointClouds."""
        return type(self)(np.concatenate([self.arr, other.arr], axis=1))

    """ Constructor methods """
 
    @classmethod
    def from_las(cls, *fpaths):
        """Initialise PointCloud from one or more .las files.
    
        Arguments
        ---------
        fpaths: str
            filepaths of .las file containing 3D point coordinates
       
        """
        if len(fpaths) > 1:
           return cls(_combine_las(*fpaths))
        else:
            return cls(_get_las_xyz(fpaths[0]))

    @classmethod
    def from_tiles(cls, bounds, *fpaths):
        """Initialise PointCloud conforming to bounds from one or more .las files.

        Arguments
        ---------
        bounds: tuple or `Bounds` namedtuple
            (minx, miny, minz, maxx, maxy, maxz) bounds of tile
        fpaths: str
            filepaths of .las file containing 3D point coordinates
        
        """
        bounds = Bounds(*bounds)
        # Determine which tiles intersect bounds
        tiles = [fpath for fpath in fpaths
                 if _intersects_3D(bounds, _get_las_bounds(fpath))] 
        return cls.from_las(*tiles).crop(*bounds) 

    @classmethod
    def from_laspy_File(cls, f):
        """Initialise PointCloud from a laspy File.
        
        Arguments
        ---------
        f: `laspy.file.File` instance
            file object must be open, and will remain so
        
        """
        return cls((f.x, f.y, f.z), header=f.header.copy())

    @classmethod
    def from_txt(cls, *fpaths):
        """Initialise PointCloud from a plaintext file.

        Arguments
        ---------
        fpaths: str
            filepath of an ASCII 3-column (xyz) whitespace-delimited
            .txt (aka .xyz) file
       
        """
        if len(fpaths) > 1:
            raise NotImplementedError
        else:
            return cls(np.loadtxt(*fpaths).T)

    @classmethod
    def from_None(cls):
        """Initialise an empty PointCloud."""
        return cls(None)

    """ Instance methods """
    @property
    def x(self):
        """The x dimension of point coordinates."""
        return self.arr[0]

    @property
    def y(self):
        """The y dimension of point coordinates."""
        return self.arr[1]

    @property
    def z(self):
        """The z dimension of point coordinates."""
        return self.arr[2]

    @property
    def points(self):
        """Get point coordinates as a structured n*3 array).
        
        Returns
        -------
        structured np.ndarray containing 'x', 'y' and 'z' point coordinates
    
        """
        return self.arr.T.ravel().view(
               dtype=[('x', self.dtype), ('y', self.dtype), ('z', self.dtype)])

    @property
    def bounds(self):
        """Boundary box surrounding PointCloud.
        
        Returns
        -------
        namedtuple (minx, miny, minz, maxx, maxy, maxz)
        
        """
        x,y,z = self.arr
        return Bounds(x.min(), y.min(), z.min(),
                      x.max(), y.max(), z.max())


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
    
    def crop(self, minx=None, miny=None, minz=None,
                   maxx=None, maxy=None, maxz=None,
                   return_empty=False):
        """Crop point cloud to (lower-inclusive, upper-exclusive) bounds.
        
        Arguments
        ---------
        minx, miny, minz, maxx, maxy, maxz: float or int (default: None)
            minimum and maximum bounds to crop pointcloud to within
            None results in no cropping at that bound
        return_empty: bool (default: False)
            whether to allow empty pointclouds to be created or raise an
            EmptyPointCloud exception        
        
        Returns
        -------
        PointCloud instance
            new object containing only points within specified bounds
        
        """
        bounds = Bounds(minx, miny, minz, maxx, maxy, maxz)
        oob = are_out_of_bounds(self, bounds)
        
        # Deal with empty pointclouds
        if oob.all():
            if return_empty:
                return type(self)(None)
            else:
                raise EmptyPointCloud, "No points in crop bounds:\n{}".format(
                                        bounds)
         
        return type(self)(self.arr[:, ~oob])

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
            f.x, f.y, f.z = self.arr

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
        idx = np.random.choice(len(self), n, replace=False)
        return type(self)(self.arr[:, idx])


    def merge(self, *pointclouds):
        """Merge this pointcloud with other instances.
        
        Arguments
        ---------
        *pointclouds: `PointCloud` instances
        
        """
        return merge(type(self), self, *pointclouds)

class Bounds(namedtuple('Bounds', ['minx', 'miny', 'minz', 'maxx', 'maxy', 'maxz'])):
    """`namedtuple` describing the bounds box surrounding PointCloud."""
    __slots__ = ()
    _format = '{:.3g}'

    def __new__(cls, minx, miny, minz, maxx, maxy, maxz):
        """Create new instance of Bounds(minx, miny, minz, maxx, maxy, maxz)
        
        Args
        ----
        minx, miny, minz, maxx, maxy, maxz: numeric or None
            minimum or maximum bounds in each dimension
            None will be coerced to -numpy.inf (mins) or numpy.inf (maxes)
        
        """
        # Coerce bounds to floats, and nones to infs
        kwargs = locals()
        for b, inf in zip(('min', 'max'),
                          (-np.inf, np.inf)):
            for dim in 'xyz':
                bound = b + dim
                value = kwargs[bound]
                kwargs[bound] = inf if value is None else float(value)
        
        kwargs.pop('cls') # must be passed positionally
        return super(cls, cls).__new__(cls, **kwargs)

    def __str__(self):
        template = ('Bounds: minx={f}, miny={f}, minz={f} \n        '
                    'maxx={f}, maxy={f}, maxz={f}'.format(f=self._format))
        return template.format(*self)


def _combine_las(*fpaths):
    """Efficiently combine las files to a single [xs, ys, zs] array."""
    sizes = {fpath: _get_las_npoints(fpath) for fpath in fpaths}
    npoints = sum(sizes.values())
    arr = np.empty((3, npoints), dtype = _DTYPE) # initialise array
    
    # Fill array piece by piece
    i = 0 # start point
    for fpath, size in sizes.iteritems():
        j = i + size # end point
        arr[:,i:j] = _get_las_xyz(fpath)
        i = j
    return arr

def _get_las_npoints(fpath):
    """Return the number of points in a .las file.
    
    Note: npoints is read from the file's header, which is not guuaranteed
          to be at all accurate. This may be a source of error.
    """
    with File(fpath) as f:
        return f.header.count

def _get_las_xyz(fpath):
    """Return [x, y, z] list of coordinate arrays from .las file."""
    with File(fpath) as f:
        return [f.x, f.y, f.z]

def _get_las_bounds(fpath):
    """Return the bounds of file at fpath."""
    with File(fpath) as f:
        return Bounds(*(f.header.min + f.header.max))

def _intersects_1D(A, B):
    """True if (min, max) tuples intersect."""
    return False if (B[1] <= A[0]) or (B[0] >= A[1]) else True

def _intersects_3D(A, B):
    """True if bounds A and B intersect."""
    return all([_intersects_1D((A[i], A[i+3]), (B[i], B[i+3]))
                for i in range(3)])

def _iter_out_of_bounds(pc, bounds):
    """Iteratively determine point coordinates outside of bounds.

    Arguments
    ---------
    pc: `PointCloud` instance
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
        yield compare(getattr(pc, c), bound)


def are_out_of_bounds(pc, bounds):
    """ Determine whether each point in pc is out of bounds
    
    Arguments
    ---------
    pc: `PointCloud` instance
    bounds: `Bounds` namedtuple
        (minx, miny, minz, maxx, maxy, maxz) to test point coordinates against
    
    Returns
    -------
    `numpy.ndarray` (shape=(len(pc),))
        bools specifying whether any of the (x, y, z) dimensions of points
        in `pc` are outside of the specified `bounds`
    
    """
    oob = np.zeros(len(pc))
    for comparison in _iter_out_of_bounds(pc, bounds):
        oob = np.logical_or(comparison, oob)
    return oob

def merge(cls, *pointclouds):
    """Return an instance of `cls` containing merged `pointclouds`.
    
    Arguments
    ---------
    cls: `PointCloud` class (or subclass)
    *pointclouds: instances of `PointCloud` (or subclass)
    """
    sizes = [len(pc) for pc in pointclouds]
    arr = np.empty((3, sum(sizes)), dtype=_DTYPE)
    
    # Build up array from pcs
    i = 0
    for pc, size in zip(pointclouds, sizes):
        j = i + size
        arr[:,i:j] = pc.arr
        i = j
    return cls(arr)
