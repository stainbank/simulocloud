import pytest
import test_pointcloud
import numpy as np
import itertools
import simulocloud.pointcloud
import simulocloud.tiles
import test_pointcloud

""" fixtures """

@pytest.fixture
def tile():
    """A simple, immutable `Tile`, subclass of `PointCloud`."""
    return simulocloud.tiles.Tile(test_pointcloud._INPUT_DATA)

@pytest.fixture
def pcs(pc_las):
    """List of overlapping pointclouds made from splitting of a .las file."""
    return test_pointcloud.overlap_pcs([pc_las], nx=4, ny=3, overlap=0.5) 

@pytest.fixture
def splitlocs(pcs):
    """Regular locations to split pointcloud in x and y; no splitting in z."""
    bounds = simulocloud.pointcloud.merge_bounds((pc.bounds for pc in pcs))
    return simulocloud.tiles.fractional_splitlocs(bounds, nx=6, ny=5, nz=None)

@pytest.fixture
def grid(pcs, splitlocs):
    """Construct a `TilesGrid` instance."""
    return simulocloud.tiles.TilesGrid(pcs, splitlocs)

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

def test_pointcloud_retiling_preserves_points(pc_las, splitlocs):
    """Does `retile` maintain the points of a single input pointcloud?"""
    tiles = simulocloud.tiles.retile([pc_las], splitlocs)
    
    assert test_pointcloud.same_len_and_bounds(np.sum(tiles), pc_las)

def test_pointcloud_retiling_obeys_splitlocs(splitlocs, grid):
    """Does `retile` split overlapping pointclouds along the specified locations?"""
    # Test that tiles are within bounds in each dimension
    for axis, d in enumerate('xyz'):
        for pcs_2d in np.swapaxes(grid.tiles, 2, axis): #iterate through axis last
            for pcs in pcs_2d:
                for i, dloc in enumerate(splitlocs[d]):
                    # Check split location divides adjacent pointclouds
                    for pc, b, compare in zip(pcs[i:i+2],
                                             ('max', 'min'),
                                             (np.less, np.greater_equal)):
                        try:
                            bound = getattr(pc.bounds, b+d)
                            assert compare(bound, dloc)
                        except simulocloud.exceptions.EmptyPointCloud:
                            pass

def test_shapes_of_edges_grid_and_tiles_grid_align(grid):
    """Does the edges array have one extra element per axis than the tile array?"""
    assert grid.edges.shape[3] == 3 # x,y,z
    for n_tile, n_bounds in zip(grid.tiles.shape, grid.edges.shape[:3]):
        assert n_bounds == n_tile+1

def test_edges_grid_describes_bounds_of_tile_grid(grid):
    """Does the grid returned by `make_edges_grid` describe that of `retile`?"""
    for ix, iy, iz in itertools.product(*map(xrange, grid.tiles.shape)):
        # Ensure pointcloud bounds fall within edges
        tile = grid.tiles[ix, iy, iz]
        for compare, edges, bounds in zip(
                (np.less_equal, np.greater_equal), # both edges inclusive due to outermost edges
                (grid.edges[ix, iy, iz], grid.edges[ix+1, iy+1, iz+1]),
                (tile.bounds[:3], tile.bounds[3:])): # mins, maxs
            for edge, bound in zip(edges, bounds):
                assert compare(edge, bound)
