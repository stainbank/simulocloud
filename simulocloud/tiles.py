"""
tile
"""
import numpy as np
import simulocloud.pointcloud
import simulocloud.exceptions

class Tile(simulocloud.pointcloud.PointCloud):
    """An immmutable pointcloud"""
    def __init__(self, xyz, header=None):
        """."""
        super(Tile, self).__init__(xyz, header)
        self._arr.flags.writeable = False
    
    @property
    def arr(self):
        """Get, but not set, the underlying (x, y, z) array of point coordinates."""
        return self._arr
    
    @arr.setter
    def arr(self, value):
        raise simulocloud.exceptions.TileException("Tile pointcloud cannot be modified")

class Tiles(object):
    """Container for tiles grid."""
    def __init__(self):
        raise NotImplemented

def retile(pcs, splitlocs, pctype=simulocloud.PointCloud):
    """Return a 3D grid of (merged) pointclouds split in x, y and z dimensions.
    
    Arguments
    ---------
    pcs: seq of `simulocloud.pointcloud.PointCloud`
    splitlocs: dict
        {d: dlocs, ...}, where:
            d: str
                'x', 'y' and/or 'z' dimension
            dlocs: list
                locations along specified axis at which to split
                (see docs for `PointCloud.split`)
        dimensions can be omitted, resulting in no splitting in that dimension
    pctype: subclass of `simulocloud.pointcloud.PointCloud`
       type of pointclouds to return (`simulocloud.pointcloud.PointCloud`)
    
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
    bounds: `simulocloud.pointcloud.Bounds` (or similiar)
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
    bounds = simulocloud.pointcloud.Bounds(*bounds) #should be a strict bounds (min<max, etc)
    nsplits = {d: n for d, n in zip('xyz', (nx, ny, nz)) if n is not None}
    # Build splitlocs
    splitlocs = {}
    for d, nd in nsplits.iteritems():
        mind, maxd = simulocloud.pointcloud._get_dimension_bounds(bounds, d)
        splitlocs[d] = np.linspace(mind, maxd, num=nd,
                                   endpoint=False)[1:] # "inside" edges only
    
    return splitlocs

def make_edges_grid(bounds, splitlocs):
    """Return coordinate array describing the edges between retiled pointclouds.
    
    Arguments
    ---------
    bounds: `simulocloud.pointcloud.Bounds` or similiar
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
        mind, maxd = simulocloud.pointcloud._get_dimension_bounds(bounds, d)
        dlocs = np.array(splitlocs.setdefault(d, np.array([])))
        edges.append(np.concatenate([[mind], dlocs, [maxd]]))
    
    # Grid edge coordinates
    grids = np.meshgrid(*edges, indexing='ij')
    grids = [DD[..., np.newaxis] for DD in grids] # promote to 4D
    return np.concatenate(grids, axis=3)
