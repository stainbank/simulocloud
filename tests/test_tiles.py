import pytest
import test_pointcloud
import numpy as np
import itertools
import simulocloud.pointcloud
import simulocloud.tiles
import simulocloud.exceptions
import test_pointcloud

""" fixtures """

@pytest.fixture
def tile():
    """A simple, immutable `Tile`, subclass of `PointCloud`."""
    return simulocloud.tiles.Tile(test_pointcloud._INPUT_DATA)

@pytest.fixture
def pcs(pc_las):
    """List of *overlapping* pointclouds made from splitting of a .las file."""
    return test_pointcloud.overlap_pcs([pc_las], nx=4, ny=3, overlap=0.5) 

@pytest.fixture
def bounds(pcs):
    """The merged bounds of pcs."""
    return simulocloud.pointcloud.merge_bounds((pc.bounds for pc in pcs))

@pytest.fixture
def splitlocs(pcs, bounds):
    """Regular locations to split pointcloud in x and y; no splitting in z."""
    return simulocloud.tiles.fractional_splitlocs(bounds, nx=6, ny=5, nz=None)

@pytest.fixture
def edges(pcs, splitlocs, bounds):
    """The edges array defined by splitlocs."""
    return simulocloud.tiles.make_edges(bounds, splitlocs)

@pytest.fixture
def tiles(pcs, edges):
    """Pointclouds gridded into edges."""
    return simulocloud.tiles.grid_pointclouds(pcs, edges)

@pytest.fixture
def grid(tiles, edges):
    """Construct a `TilesGrid` instance."""
    return simulocloud.tiles.TilesGrid(tiles, edges, validate=False)

@pytest.fixture
def half_indices(grid):
    """Tuple of ints specifying the halfway (rounding down) indices of grid tiles."""
    return tuple((i//2 for i in grid.shape))

""" tests """
def test_tile_array_is_immutable(tile):
    """Is an error raised when trying to modify the coordinates of a `Tile`?"""
    with pytest.raises(ValueError):
        tile.arr += 2.
    with pytest.raises(ValueError):
        tile.arr[0,0] = 1.

def test_tile_array_cannot_be_changed(tile):
    """Is an error raised when trying to set to the coordinates of a `Tile`?"""
    arr2 = tile.arr*2
    with pytest.raises(simulocloud.exceptions.TileException):
        tile.arr = arr2

def test_shapes_of_edges_and_tiles_grid_align(grid):
    """Does the edges array have one extra element per axis than the tile array?"""
    assert grid.edges.shape[3] == 3 # x,y,z
    for n_tile, n_bounds in zip(grid.tiles.shape, grid.edges.shape[:3]):
        assert n_bounds == n_tile+1

def test_edges_describes_bounds_of_tile_grid(grid):
    """Does the grid returned by `make_edges` describe that of `grid_pointclouds`?"""
    for ix, iy, iz in itertools.product(*map(xrange, grid.tiles.shape)):
        # Ensure pointcloud bounds fall within edges
        tile = grid.tiles[ix, iy, iz]
        for compare, edges, bounds in zip(
                (np.less_equal, np.greater_equal), # both edges inclusive due to outermost edges
                (grid.edges[ix, iy, iz], grid.edges[ix+1, iy+1, iz+1]),
                (tile.bounds[:3], tile.bounds[3:])): # mins, maxs
            for edge, bound in zip(edges, bounds):
                assert compare(edge, bound)

def test_TilesGrid_is_self_validating(grid):
    """The result of this test should be identical to that of `test_edges_describes_bounds_of_tile_grid`."""
    assert grid and grid.validate()
    grid.edges *= 100
    assert not grid.validate()

def test_TilesGrid_initialisation_fails_if_invalid(pcs, edges, tiles):
    """Is a `TilesGridException` raised when trying to create a `TilesGrid` instance with edges which incorrectly describe tiles?."""
    # Make edges out of range
    edges = edges + 100
    
    with pytest.raises(simulocloud.exceptions.TilesGridException):
        simulocloud.tiles.TilesGrid(tiles, edges)

def test_pointcloud_retiling_obeys_splitlocs(pcs, splitlocs):
    """Does the tiles grid created by `from_splitlocs` adhere to the specified split locations?"""
    grid = simulocloud.tiles.TilesGrid.from_splitlocs(pcs, splitlocs)
    
    # Test that tiles are within bounds in each axis
    for i, axis in enumerate('xyz'):
        for pcs_2d in np.swapaxes(grid.tiles, 2, i): #iterate through axis last
            for pcs in pcs_2d:
                for j, loc in enumerate(splitlocs[axis]):
                    # Check split location divides adjacent pointclouds
                    for pc, b, compare in zip(pcs[j:j+2],
                                             ('max', 'min'),
                                             (np.less, np.greater_equal)):
                        try:
                            bound = getattr(pc.bounds, b+axis)
                            assert compare(bound, loc)
                        except simulocloud.exceptions.EmptyPointCloud:
                            pass

def test_retiling_is_upper_bounds_exclusive(pcs, tiles):
    """Are points lost when gridded to edges whose bounds are equal to pointcloud bounds?"""
    pc = simulocloud.pointcloud.merge(pcs)
    pc_tiles = simulocloud.pointcloud.merge(tiles.flatten())
    assert pc_tiles.bounds != pc.bounds and len(pc_tiles) < len(pc)

def test_retiling_is_optionally_upper_bounds_inclusive(pcs, bounds, splitlocs):
    """Can points be preserved when gridding with pointcloud bounds?"""
    edges = simulocloud.tiles.make_edges(bounds, splitlocs, inclusive=True)
    tiles = simulocloud.tiles.grid_pointclouds(pcs, edges)
    
    pc = simulocloud.pointcloud.merge(pcs)
    pc_tiles = simulocloud.pointcloud.merge(tiles.flatten())
    assert test_pointcloud.same_len_and_bounds(pc_tiles, pc)

def test_TilesGrid_is_subsettable(grid, half_indices):
    """Does a `TilesGrid` return a subset of itself when indexed?."""
    ix, iy, iz = half_indices
    subset = grid[ix:, iy:, iz:]
    assert subset and subset.validate()

def test_TilesGrid_subset_with_integers_has_arrays(grid, half_indices):
    """Are full sized tile and edge arrays produced by integer subsetting?."""
    ix, iy, iz = half_indices
    subset = grid[ix, iy, iz] # single element
    assert (subset.validate()) and (subset.tiles.shape == (1,1,1))

def test_TilesGrid_indexing_accepts_negative_indices(grid):
    """Can negative (i.e. backwards) indexing be used?"""
    shape = np.array(grid.shape)
    last = shape-1
    neg = last - shape
    
    grid_last = grid[tuple(last)]
    grid_neg = grid[tuple(neg)]
    
    assert (grid_last.bounds == grid_neg.bounds) and (grid_last.tiles == grid_neg.tiles)

def test_subsetting_to_empty_is_reasonable(grid):
    """Are both tiles and edges empty when subsetting TilesGrid to empty?"""
    ix, iy, iz = grid.tiles.shape
    subset = grid[ix:, iy:, iz:]
    assert (not subset) and (not subset.edges.size)

def test_TilesGrid_indexing_doesnt_accept_steps(grid, half_indices):
    """Is a ValueError raised when attempting to create a non-contiguous or negatively indexed subset of TilesGrid?"""
    ix, iy, iz = half_indices
    for start, stop, step in ((0, ix, 2),(ix, 0, -2)):
        with pytest.raises(ValueError):
            grid[start:stop:step]

def test_TilesGrid_is_iterable(grid):
    """Does a for loop run over a TilesGrid instance terminate?"""
    n = len(grid)
    for i, pc in enumerate(grid):
        if i == n:
            assert 0
    else:
        assert i == n-1 # visited all the pointclouds

def test_default_regular_edges_preserves_bounds(bounds):
    """Does the default creation of a regularly spaced `edges` array have exactly `bounds`?."""
    # random spacings
    spacings = {axis: np.random.rand() for axis in 'xy'}
    edges = simulocloud.tiles.make_regular_edges(bounds, spacings)
    edge_bounds = simulocloud.pointcloud.Bounds(*
            np.concatenate([edges[0,0,0], edges[-1,-1,-1]]))
    # Ensure even spacing
    for axis, sl in zip('xy', ((slice(None),0,0,0), (0,slice(None),0,1))):
        axedges = edges[sl]
        axspacings = axedges[1:] - axedges[:-1]
        assert np.allclose(axspacings, axspacings[0])
    assert edge_bounds == bounds
