"""
pointcloud

Read in and store point clouds.
"""

import numpy as np
import string
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
        # Determine which tiles intersect bounds
        tiles = [fpath for fpath in fpaths
                 if _intersects_3D(InfBounds(*bounds), _get_las_bounds(fpath))] 
        return cls.from_las(*tiles).crop(bounds) 

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
        
        Raises
        ------
        EmptyPointCloud
            if there are no points
        """
        x,y,z = self.arr
        try:
            return Bounds(x.min(), y.min(), z.min(),
                          x.max(), y.max(), z.max())
        except ValueError:
            raise EmptyPointCloud, "len 0 PointCloud has no Bounds"

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
    
    def crop(self, bounds, destructive=False, allow_empty=False):
        """Crop point cloud to (lower-inclusive, upper-exclusive) bounds.
        
        Arguments
        ---------
        bounds: `Bounds` namedtuple
            (minx, miny, minz, maxx, maxy, maxz) to test point coordinates against
            None results in no cropping at that bound
        destructive: bool (default: False)
            whether to remove cropped values from pointcloud
        allow_empty: bool (default: False)
            whether to allow empty pointclouds to be created or raise an
            EmptyPointCloud exception        
        
        Returns
        -------
        PointCloud instance
            new object containing only points within specified bounds
        
        """
        bounds = Bounds(*bounds)
        oob = are_out_of_bounds(self, bounds)
        # Deal with empty pointclouds
        if oob.all():
            if allow_empty:
                return type(self)(None)
            else:
                raise EmptyPointCloud, "No points in crop bounds:\n{}".format(
                                        bounds)
         
        cropped = type(self)(self.arr[:, ~oob])
        if destructive:
            self.__init__(self.arr[:, oob])
        return cropped

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

    def split(self, axis, dlocs, pctype=None, allow_empty=True):
        """Split this pointcloud at specified locations along axis.
        
        Arguments
        ---------
        axis: str
            point coordinate component ('x', 'y', or 'z') to split along
        dlocs: iterable of float
            points along `axis` at which to split pointcloud
        pctype: subclass of `PointCloud`
           type of pointclouds to return
        allow_empty: bool (default: True)
            whether to allow empty pointclouds to be created or raise an
            EmptyPointCloud exception

        Returns
        -------
        pcs: list of `pctype` (PointCloud) instances
            pointclouds with `axis` bounds defined sequentially (low -> high)
            by self.bounds and dlocs
        """
        # Copy pointcloud
        if pctype is None:
            pctype = type(self)
        pc = pctype(self.arr)
        
        # Sequentially (high -> low) split pointcloud
        none_bounds = Bounds(*(None,)*6)
        pcs = [pc.crop(none_bounds._replace(**{'min'+axis: loc}),
                       destructive=True, allow_empty=allow_empty)
                  for loc in sorted(dlocs)[::-1]]
        pcs.append(pc)
        
        return pcs[::-1]


class NoneFormatter(string.Formatter):
    """Handle an attempt to apply decimal formatting to `None`.

    `__init__` and `get_value` are from https://stackoverflow.com/a/21664626
    and allow autonumbering.
    """

    def __init__(self):
        super(NoneFormatter, self).__init__()
        self.last_number = 0

    def get_value(self, key, args, kwargs):
        if key == '':
            key = self.last_number
            self.last_number += 1
        return super(NoneFormatter, self).get_value(key, args, kwargs)

    def format_field(self, value, format_spec):
        """Format any `None` value sans specification (i.e. default format)."""
        if value is None:
            return format(value)
        else:
            return super(NoneFormatter, self).format_field(value, format_spec)
            if value is None:
                return format(value)
            else: raise e


class Bounds(namedtuple('Bounds', ['minx', 'miny', 'minz',
                                   'maxx', 'maxy', 'maxz'])):
    """`namedtuple` describing the bounds box surrounding PointCloud."""
    __slots__ = ()
    _format = '{:.3g}'

    def __str__(self):
        """Truncate printed values as specified by class attribute `_format`."""
        template = ('Bounds: minx={f}, miny={f}, minz={f}\n        '
                    'maxx={f}, maxy={f}, maxz={f}'.format(f=self._format))
        # Formatter must be recreated each time to reset value counter
        return NoneFormatter().format(template, *self)

class InfBounds(Bounds):
    """`Bounds` namedtuple, with `None`s coerced to `inf`s."""
    __slots__ = ()

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
        yields, for each bound of lower, upper of x, y, z, not equal to `None`,
        a boolean numpy.ndarray describing whether each point falls outside of
        that bound in that dimension
    
    Notes
    -----
    Comparisons are python-like, i.e.:
        x < minx
        x >= maxx
    Comparisons to `None` are skipped (generator will be empty if all bounds
    are `None`)
    """
    for i, dim in enumerate(pc.arr):
        for compare, bound in zip((np.less, np.greater_equal),
                                  (bounds[i], bounds[i+3])):
            if bound is not None:
                yield compare(dim, bound)

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
    oob = np.zeros(len(pc), dtype=bool)
    for comparison in _iter_out_of_bounds(pc, bounds):
        oob = np.logical_or(comparison, oob)
    return oob

def _get_dimension_bounds(pc, d):
    """Return the (min, max) of dimension `d` in bounds of PointCloud (or Bounds namedtuple) `pc`."""
    try:
        bounds = pc.bounds
    except AttributeError:
        bounds = pc
    
    return tuple([getattr(bounds, b + d) for b in ('min', 'max')])

def merge_bounds(bounds):
    """Find overall bounds of pcs (or bounds).
    
    Arguments
    ---------
    pcs: iterable of `Bounds` namedtuple (or similiar)
         None values will be treated as appropriate inf
    
    Returns
    -------
    `Bounds` namedtuple
        describing total area covered by args
    
    """
    # Coerce Nones to Infs
    all_bounds = [InfBounds(*bounds) for bounds in bounds]
    
    # Extract mins/maxs of dimensions
    all_bounds = np.array(all_bounds)
    return Bounds(all_bounds[:,0].min(), all_bounds[:,1].min(), all_bounds[:,2].min(),
                  all_bounds[:,3].max(), all_bounds[:,4].max(), all_bounds[:,5].max())

"""PointCloud manipulation"""

def merge(pctype, *pointclouds):
    """Return an instance of `pctype` containing merged `pointclouds`.
    
    Arguments
    ---------
    pctype: `PointCloud` class (or subclass)
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
    return pctype(arr)

def retile(pcs, splitlocs, pctype=PointCloud):
    """Return a 3D grid of (merged) pointclouds split in x, y and z dimensions.
    
    Arguments
    ---------
    pcs: seq of `PointCloud`
    splitlocs: dict
        {d: dlocs, ...}, where:
            d: str
                'x', 'y' and/or 'z' dimension
            dlocs: list
                locations along specified axis at which to split
                (see docs for `PointCloud.split`)
        dimensions can be omitted, resulting in no splitting in that dimension
    pctype: subclass of `PointCloud` (default=`PointCloud`)
       type of pointclouds to return
    
    Returns
    -------
    tile_grid: `numpy.ndarray` (ndim=3, dtype=object)
        3D array containing pointclouds (of type `pctype`) resulting from the
        (collective) splitting of `pcs` in each dimension according to `dlocs`
        in `splitlocs`
        sorted `dlocs` align with sequential pointclouds along each array axis:
            0:x, 1:y, 2:z
    
    """
    shape = [] #nx, ny, nz
    for d in 'x', 'y', 'z':
        dlocs = sorted(splitlocs.setdefault(d, []))
        shape.append(len(dlocs) + 1) #i.e. n pointclouds created by split
        #! Should assert splitlocs within bounds of pcs
        splitlocs[d] = dlocs
    
    # Build 4D array with pcs split in x, y and z
    tile_grid = np.empty([len(pcs)] + shape, dtype=object)
    for i, pc in enumerate(pcs):
        pcs = pc.split('x', splitlocs['x'], pctype=pctype)
        for ix, pc in enumerate(pcs):
            pcs = pc.split('y', splitlocs['y'])
            for iy, pc in enumerate(pcs):
                pcs = pc.split('z', splitlocs['z'])
                # Assign pc to predetermined location
                for iz, pc in enumerate(pcs):
                    tile_grid[i, ix, iy, iz] = pc
    
    # Flatten to 3D
    return np.sum(tile_grid, axis=0)

def fractional_splitlocs(bounds, nx=None, ny=None, nz=None):
    """Generate locations to split bounds into n even sections per axis.
    
    Arguments
    ---------
    bounds: `Bounds` namedtuple (or similiar)
        bounds within which to create tiles
    nx, ny, nz : int (default=None)
        number of pointclouds desired along each axis
        no splitting if n < 2 (or None)
    
    Returns
    -------
    splitlocs: dict ({d: dlocs, ...)}
        lists of locations for each dimension d (i.e. 'x', 'y', 'z')
        len(dlocs) = nd-1; omitted if nd=None
     
    """
    bounds = Bounds(*bounds) #should be a strict bounds (min<max, etc)
    nsplits = {d: n for d, n in zip('xyz', (nx, ny, nz)) if n is not None}
    # Build splitlocs
    splitlocs = {}
    for d, nd in nsplits.iteritems():
        mind, maxd = _get_dimension_bounds(bounds, d)
        splitlocs[d] = np.linspace(mind, maxd, num=nd,
                                   endpoint=False)[1:] # "inside" edges only
    
    return splitlocs

def make_edges_grid(bounds, splitlocs):
    """Return coordinate array describing the edges between retiled pointclouds.
    
    Arguments
    ---------
    bounds: tuple or `Bounds` namedtuple
       (minx, miny, minz, maxx, maxy, maxz) bounds of entire grid
    splitlocs: dict {d: dlocs, ...}
        same as argument to `retile`
    
    Returns
    -------
    edges_grid: `numpy.ndarray` (ndim=4, dtype=float)
        4D array containing x, y and z coordinate arrays (see documentation for
        `numpy.meshgrid`), indexed by 'ij' and concatenated in 4th dimension
        indices, such that `edges_grid[ix, iy, iz, :]` returns a single point
        coordinate in the form `array([x, y, z])`
    
    Notes and Examples
    ------------------
    This function is intended to be used alongside `retile` with the same
    `splitlocs` and `bounds` equal to those of the pointcloud (or merged bounds
    of pointclouds) to be retiled. The resultant `edges` grid provides a
    spatial description of the pointclouds in the `tiles` grid:
    - the coordinates at `edges[ix, iy, iz]` lies between the two adjacent
      pointclouds `tiles[ix-1, iy-1, iz-1], tiles[ix, iy, iz]`
    - `edges[ix, iy, iz]` and `edges[ix+1, iy+1, iz+1]` combine to form a set
      of bounds which contain --- but are not (necessarily) equal to --- those
      of the pointcloud at `tile_grid[ix, iy, iz]`
     
    >>> splitlocs = fractional_splitlocs(pc.bounds, nx=10, ny=8, nz=5)
    >>> tiles = retile(pc, splitlocs)
    >>> edges = make_edges_grid(pc.bounds, splitlocs)
    >>> print tiles.shape, edges.shape # +1 in each axis
    (10, 8, 5) (11, 9, 6, 3)
    >>> ix, iy, iz = 5, 3, 2
    # Show edge between tile pointclouds
    >>> print (tiles[ix-1, iy-1, iz-1].bounds[3:], # upper bounds
    ...        edges[ix, iy, iz],
    ...        tiles[ix, iy, iz].bounds[:3]) # lower bounds
    ((14.99, 24.98, 1.98), array([ 15.,  25.,   2.]), (15.01, 25.02, 2.09))
    >>> # Show bounds around tile
    >>> print tiles[ix, iy, iz].bounds
    Bounds: minx=15, miny=25, minz=2.09
            maxx=16, maxy=26.2, maxz=2.99
    >>> print Bounds(*np.concatenate([edges[ix, iy, iz],
    ...                               edges[ix+1, iy+1, iz+1]]))
    Bounds: minx=15, miny=25, minz=2
            maxx=16, maxy=26.2, maxz=3

    """
    #! Should fail if splitlocs is not within bounds

    # Determine bounds for each tile in each dimension
    edges = []
    for d in 'xyz':
        d_edges = []
        mind, maxd = _get_dimension_bounds(bounds, d)
        dlocs = np.array(splitlocs.setdefault(d, np.array([])))
        edges.append(np.concatenate([[mind], dlocs, [maxd]]))
    
    # Grid edge coordinates
    grids = np.meshgrid(*edges, indexing='ij')
    grids = [DD[..., np.newaxis] for DD in grids] # promote to 4D
    return np.concatenate(grids, axis=3)
