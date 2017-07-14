"""
tiles
"""
import numpy as np
import itertools
import simulocloud.pointcloud
import simulocloud.exceptions

class Tile(simulocloud.pointcloud.PointCloud):
    """An immmutable pointcloud."""
    def __init__(self, xyz, header=None):
        """See documentation for `simulocloud.pointcloud.Pointcloud`."""
        super(Tile, self).__init__(xyz, header)
        self._arr.flags.writeable = False
    
    @property
    def arr(self):
        """Get, but not set, the underlying (x, y, z) array of point coordinates."""
        return self._arr
    
    @arr.setter
    def arr(self, value):
        raise simulocloud.exceptions.TileException("Tile pointcloud cannot be modified")

class TilesGrid(object):
    """Container for grid of tiles described spatially by edges grid.
    
    Attributes
    ----------
    tiles: `numpy.ndarray` (ndim=3, dtype=object)
        spatially contiguous pointclouds (usually type `Tile`) gridded to a 3D
        array ordered by sequence of intervals in x (0), y (1) and z (2)
    edges: `numpy.ndarray` (ndim=4, dtype=float)
        three 3D x, y and z coordinate arrays (i,j indexing) concatenated in
        4th axis, defining intervals seperating elements in `tiles` such that:
        - `edges[ix, iy, iz]` returns a point coordinate at the corner between
          adjacent pointclouds `tiles[ix-1, iy-1, iz-1], tiles[ix, iy, iz]`
        - the bounds produced by concatenation of `edges[ix, iy, iz]` and
          `edges[ix+1, iy+1, iz+1]` (i.e. `grid[ix, iy, iz].bounds`)
          are guaranteed to spatially contain (but not necessarily equal) those
          of the pointcloud at `tiles[ix, iy, iz]`
    bounds: `Bounds`
        defined by the outermost coordinates of `edges`
    
    Subsetting
    ----------
    A `TilesGrid` can be sliced or indexed to produce a subset (i.e. another
    `TilesGrid` instance), with the following restrictions:
    - step size must be 1 (or None)
    - negative steps (reverse slicing) is unsupported
    
    Subsetting produces views into, not copies of, the `tiles` and `edge` grid
    arrays of the parent. This makes subsetting a light operation, but care
    must be taken not to modify these attributes.
    
    """
    def __init__(self, tiles, edges, validate=True):
        """Directly initialise `TilesGrid` from grids.
        
        Arguments
        ---------
        tiles: `numpy.ndarray` (ndim=3, dtype=object)
            3D array of ordered pointclouds gridded onto `edges`
            usually produced by `grid_pointclouds`
        edges: `numpy.ndarray` (ndim=4, dtype=float)
            4D array of shape (nx+1, ny+1, nz+1, 3) where nx, ny, nz = tiles.shape
            usually produced by `make_edges`
        
        Instantiation by constructor classmethods is preferred.
        
        """
        self.tiles = tiles
        self.edges = edges
        if validate:
            if not self.validate():
                msg = "Tiles do not fit into edges grid"
                raise simulocloud.exceptions.TilesGridException(msg)

    def __getitem__(self, key):
        """Return a subset of TilesGrid instance using numpy-like indexing.
        
        Notes
        -----
        - Steps are forbidden; only contiguous TilesGrids can be created
        - Negative steps are forbidden
        
        """
        # Coerce key to list
        try:
            key = list(key)
        except TypeError:
            key = [key]
        
        # Freeze slice indices to shape of tiles array
        key_ = []
        for sl, nd in itertools.izip_longest(key, self.tiles.shape,
                                             fillvalue=slice(None)):
            try: # assume slice
                start, stop, step = sl.indices(nd)
            except AttributeError: # coerce indices to slice
                if sl is None:
                    start, stop, step = slice(None).indices(nd)
                else: # single element indexing 
                    stop = None if sl == -1 else sl+1
                    start, stop, step = slice(sl, stop).indices(nd)
            
            if not step == 1:
                raise ValueError("TilesGrid must be contiguous, slice step must be 1")
            
            key_.append(slice(start, stop))
         
        # Extend slice stops by 1 for edges array
        ekey = [slice(sl.start, sl.stop+1) if sl.stop - sl.start
                else slice(sl.start, sl.stop) # dont create edges where no tiles
                for sl in key_]
        
        return type(self)(self.tiles[key_], self.edges[ekey], validate=False)

    def __iter__(self):
        """Iterate over the tiles array."""
        return np.nditer(self.tiles, flags=["refs_ok"])

    def __len__(self):
        """Return the number of elements in tiles grid."""
        return self.tiles.size

    def __nonzero__(self):
        """Return True if there are any tiles."""
        return bool(len(self))

    @classmethod
    def from_splitlocs(cls, pcs, splitlocs, inclusive=True):
        """Construct `TilesGrid` instance by retiling pointclouds.
        
        Arguments
        ---------
        pcs: seq of `simulocloud.pointcloud.Pointcloud`
        splitlocs: dict {axis: locs, ...}, where:
            axis: str
                'x', 'y' and/or 'z'
            locs: list
                locations along specified axis at which to split
                (see docs for `simulocloud.pointcloud.PointCloud.split`)
        
            axes can be omitted, resulting in no splitting in that
            axis
        inclusive: bool (optional, default=True)
            if True, upper bounds of grid outer edges are increased by 1e.-6,
            so that all points in `pcs` are preserved upon gridding
            if False, any points exactly on the upper bounds of `pcs` are lost
            (i.e. maintain upper bounds exclusive cropping)
        
        Returns
        -------
        `TilesGrid` instance
            internal edges defined by `splitlocs`
            lower grid bounds are equal to merged bounds of `pcs`, upper grid
            bounds are 1e-6 higher than those of `pcs` if `inclusive` is True,
            otherwise they are equal
        
        """
        # Sort splitlocs and determine their bounds
        mins, maxs = [],[]
        for axis in 'xyz':
            locs = sorted(splitlocs.get(axis, []))
            try:
                min_, max_ = locs[0], locs[-1]
            except IndexError:
                min_, max_ = np.inf, -np.inf # always within another bounds
            splitlocs[axis] = locs
            mins.append(min_), maxs.append(max_)
        
        # Ensure grid will be valid
        splitloc_bounds = simulocloud.pointcloud.Bounds(*(mins + maxs))
        pcs_bounds = simulocloud.pointcloud.merge_bounds([pc.bounds for pc in pcs])
        if not simulocloud.pointcloud._inside_bounds(splitloc_bounds, pcs_bounds):
            raise ValueError("Split locations must be within total bounds of pointclouds")
        
        edges = make_edges(pcs_bounds, splitlocs)
        tiles = grid_pointclouds(pcs, edges, pctype=Tile)
        
        return cls(tiles, edges, validate=False)

    @property
    def bounds(self):
        """Return the bounds containing the entire grid of tiles."""
        bounds = np.concatenate([self.edges[0,0,0], self.edges[-1,-1,-1]])
        return simulocloud.pointcloud.Bounds(*bounds)

    @property
    def shape(self):
        """Return the shape of the grid of tiles."""
        return self.tiles.shape
    
    def validate(self):
        """Return True if grid edges accurately describes tiles."""
        for ix, iy, iz in itertools.product(*map(xrange, self.tiles.shape)):
            # Ensure pointcloud bounds fall within edges
            tile = self.tiles[ix, iy, iz]
            for compare, edges, bounds in zip(
                    (np.less_equal, np.greater_equal), # both edges inclusive due to outermost edges
                    (self.edges[ix, iy, iz], self.edges[ix+1, iy+1, iz+1]),
                    (tile.bounds[:3], tile.bounds[3:])): # mins, maxs
                for edge, bound in zip(edges, bounds):
                    if not compare(edge, bound):
                        return False
        
        return True

def grid_pointclouds(pcs, edges, pctype=Tile):
    """Return a 3D array of (merged) pointclouds gridded to edges.
    
    Arguments
    ---------
    pcs: seq of `simulocloud.pointcloud.PointCloud`
    edges: `numpy.ndarray` (ndim=4, dtype=float)
        ij-indexed meshgrids for x, y and z stacked in 4th axis, whose values
        defining boundaries of cells into `pcs` will be gridded
    pctype: subclass of `simulocloud.pointcloud.PointCloud` (optional)
        type of pointclouds to return
        default = `simulocloud.pointcloud.PointCloud`
    
    Returns
    -------
    tiles: `numpy.ndarray` (ndim=3, dtype=object)
        3D array containing pointclouds (of type `pctype`) resulting from the
        (collective) splitting of `pcs` in each axis according to `locs`
        in `splitlocs`
        sorted `locs` align with sequential pointclouds along each array axis:
            0:x, 1:y, 2:z
    
    """
    # Pre-allocate empty tiles array
    shape = (len(pcs),) + tuple((n-1 for n in edges.shape[:3]))
    tiles = np.empty(shape, dtype=object)
    
    # Build 4D array with pcs split in x, y and z
    for i, pc in enumerate(pcs):
        pcs = pc.split('x', edges[:,0,0,0], pctype=pctype)[1:-1]
        for ix, pc in enumerate(pcs):
            pcs = pc.split('y', edges[0,:,0,1])[1:-1]
            for iy, pc in enumerate(pcs):
                pcs = pc.split('z', edges[0,0,:,2])[1:-1]
                # Assign pc to predetermined location
                tiles[i, ix, iy] = pcs
    
    # Flatten to 3D
    return np.sum(tiles, axis=0)

def fractional_splitlocs(bounds, nx=None, ny=None, nz=None):
    """Generate locations to split bounds into n even sections per axis.
    
    Arguments
    ---------
    bounds: `simulocloud.pointcloud.Bounds` (or similiar)
        bounds within which to create tiles
    nx, ny, nz : int (default=None)
        number of pointclouds desired along each axis
        no splitting if n < 2 (or None)
    
    Returns
    -------
    splitlocs: dict ({axis: locs, ...)}
        lists of locations for each axis (i.e. 'x', 'y', 'z')
        len(locs) = n-1; omitted if n=None
     
    """
    bounds = simulocloud.pointcloud.Bounds(*bounds) #should be a strict bounds (min<max, etc)
    nsplits = {axis: n for axis, n in zip('xyz', (nx, ny, nz)) if n is not None}
    # Build splitlocs
    splitlocs = {}
    for axis, n in nsplits.iteritems():
        min_, max_ = simulocloud.pointcloud.axis_bounds(bounds, axis)
        splitlocs[axis] = np.linspace(min_, max_, num=n,
                                   endpoint=False)[1:] # "inside" edges only
    
    return splitlocs

def make_edges(bounds, splitlocs, inclusive=False):
    """Return coordinate array describing the edges between gridded pointclouds.
    
    Arguments
    ---------
    bounds: `simulocloud.pointcloud.Bounds` or similiar
       (minx, miny, minz, maxx, maxy, maxz) bounds of entire grid
    splitlocs: dict {axis: locs, ...}, where:
        axis: str
            'x', 'y' and/or 'z'
        locs: list
            locations along specified axis at which to split
            (see docs for `simulocloud.pointcloud.PointCloud.split`)
        axes can be omitted, resulting in no splitting in that axis
    inclusive: bool (optional, default=False)
        if True, upper bounds of grid outer edges are increased by 1e.-6,
        so that all points in a pointcloud are guaranteed to be preserved upon
        gridding when `bounds` is equal to the bounds of said pointcloud
        if False, any points exactly on the upper bounds of `pcs` are lost
        (i.e. maintain upper bounds exclusive cropping)

    Returns
    -------
    edges: `numpy.ndarray` (ndim=4, dtype=float)
        4D array containing x, y and z coordinate arrays (see documentation for
        `numpy.meshgrid`), indexed by 'ij' and concatenated in 4th dimension
        indices, such that `edges[ix, iy, iz, :]` returns a single point
        coordinate in the form `array([x, y, z])`
    
    Notes and Examples
    ------------------
    An edges` grid provides a spatial description of the pointclouds in a
    `tiles` grid:
    - the coordinates at `edges[ix, iy, iz]` lies between the two adjacent
      pointclouds `tiles[ix-1, iy-1, iz-1], tiles[ix, iy, iz]`
    - `edges[ix, iy, iz]` and `edges[ix+1, iy+1, iz+1]` combine to form a set
      of bounds which contain --- but are not (necessarily) equal to --- those
      of the pointcloud at `tiles[ix, iy, iz]`
     
    >>> splitlocs = fractional_splitlocs(pc.bounds, nx=10, ny=8, nz=5)
    >>> edges = make_edges(pc.bounds, splitlocs)
    >>> tiles = grid_pointclouds([pc], edges)
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
    if inclusive:
        bounds = np.array(bounds)
        bounds[3:] += 1e-6 # expand upper bounds to ensure all points contained
        bounds = simulocloud.pointcloud.Bounds(*bounds)
     
    #! Should fail if splitlocs is not within bounds
    
    # Determine bounds for each tile in each axis
    edges = []
    for axis in 'xyz':
        axis_edges = []
        min_, max_ = simulocloud.pointcloud.axis_bounds(bounds, axis)
        locs = np.array(splitlocs.setdefault(axis, np.array([])))
        edges.append(np.concatenate([[min_], locs, [max_]]))
    
    # Grid edge coordinates
    return np.stack(np.meshgrid(*edges, indexing='ij'), axis=-1)

def make_regular_edges(bounds, spacings, bases=None, exact=False):
    """Return `edges` array with regular interval spacing.
    
    Arguments
    ---------
    bounds: `Bounds` or equivalent tuple
        (minx, miny, minz, maxx, maxy, maxz)
    spacings: `dict`
        {axis: spacing}, where
        axis: `str`
            any combination of 'x', 'y', 'z'
        spacing: numeric
            size of regular interval in that axis
            may be adjusted, see `exact` below)
    bases: `dict` (optional)
        {axis: base} to align bounds (see documentation for `align_bounds`)
        note that bounds will become unaligned if `exact` is True, unless
        `bases` and `spacings` are equal
    exact: bool (default=False):
        for a given axis, unless (maxbound-minbound)%spacing == 0, either of
        spacing or bounds must be adjusted to yield integer n intervals
        if True, upper bound will be adjusted downwards to ensure spacing is
        exactly as specified (edges bounds will no longer == `bounds`!)
        if False, spacing will be adjusted and bounds will remain as specified
    
    """
    if bases is not None:
        bounds = align_bounds(bounds, bases)
    
    splitlocs = {}
    for axis, spacing in spacings.iteritems():
        minbound, maxbound = simulocloud.pointcloud.axis_bounds(bounds, axis)
        num = int((maxbound-minbound)/(spacing*1.))
        if exact:
            # Adjust upper bound to ensure exact spacing
            maxbound = minbound + int((maxbound-minbound)/spacing) * spacing
            bounds = bounds._replace(**{'max'+axis: maxbound})
        splitlocs[axis] = np.linspace(minbound, maxbound, num=num, endpoint=False)[1:]
    
    return make_edges(bounds, splitlocs)

def align_bounds(bounds, bases):
    """Contract `bounds` such that each axis aligns on it's respective `base`.
    
    Arguments
    ---------
    bounds: `Bounds` or equivalent tuple
        (minx, miny, minz, maxx, maxy, maxz)
    bases: `dict`
        {axis: base} where
        axis: `str`
            any combination of 'x', 'y', 'z'
        base: numeric
            value onto which to align axis bounds
    
    Returns
    -------
    bounds: `Bounds`
        bounds with each each axis specified in `bases` is a multiple of the
        respective base
        always equal or smaller in area than input `bounds`
    
    Example
    -------
    >>> bounds
    Bounds(minx=3.7, miny=-11.3, minz=7.5, maxx=20.6, maxy=5.3, maxz=23.3)
    >>> bases
    {'x': 1.0, 'y': 0.5}
    >>> align_bounds(bounds, bases)
    Bounds(minx=4.0, miny=-11.0, minz=7.5, maxx=20.0, maxy=5.0, maxz=23.3)
    
    """
    bases = {axis: float(base) for axis, base in bases.iteritems()}
    bounds = simulocloud.pointcloud.Bounds(*bounds)
    
    replacements = {}
    for axis, base in bases.iteritems():
        minbound, maxbound = simulocloud.pointcloud.axis_bounds(bounds, axis)
        remain = (minbound % base)
        replacements['min'+axis] = minbound + base - remain if remain else minbound
        replacements['max'+axis] = maxbound - maxbound % base
    return bounds._replace(**replacements)
