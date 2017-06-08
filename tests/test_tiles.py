import pytest
import test_pointcloud
import numpy as np
import itertools
import simulocloud.pointcloud
import simulocloud.tiles
import test_pointcloud

""" fixtures """
@pytest.fixture
def make_grids(pc_las):
    """Create overlapping pointclouds, retile, and create associated edges."""
    pcs = test_pointcloud.overlap_pcs([pc_las], nx=4, ny=3, overlap=0.5)
    bounds = simulocloud.pointcloud.merge_bounds((pc_las.bounds for pc in pcs))
    splitlocs = simulocloud.tiles.fractional_splitlocs(bounds, nx=6, ny=5, nz=None)
    tile_grid = simulocloud.tiles.retile(pcs, splitlocs)
    edges_grid = simulocloud.tiles.make_edges_grid(bounds, splitlocs)
    
    return splitlocs, tile_grid, edges_grid
    
""" tests """
def test_pointcloud_retiling_preserves_points(pc_las):
    """Does `retile` maintain the points of a single input pointcloud?"""
    splitlocs = simulocloud.tiles.fractional_splitlocs(pc_las.bounds, nx=10, ny=20, nz=None)
    pcs_3d = simulocloud.tiles.retile([pc_las], splitlocs)
    
    assert test_pointcloud.same_len_and_bounds(np.sum(pcs_3d), pc_las)

def test_overlapping_pointclouds_retiling_obeys_splitlocs(make_grids):
    """Does `retile` split overlapping pointclouds along the specified locations?"""
    splitlocs, tile_grid, _ = make_grids
    # Test that tiles are within bounds in each dimension
    for axis, d in enumerate('xyz'):
        for pcs_2d in np.swapaxes(tile_grid, 2, axis): #iterate through axis last
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

def test_edges_grid_has_correct_shape(make_grids):
    """Does the edges array have one extra element per axis than the tile array?"""
    _, tile_grid, edges_grid = make_grids
    assert edges_grid.shape[3] == 3 # x,y,z
    for n_tile, n_bounds in zip(tile_grid.shape, edges_grid.shape[:3]):
        assert n_bounds == n_tile+1

def test_bounds_grid_describes_bounds_of_tile_grid(make_grids):
    """Does the grid returned by `make_edges_grid` describe that of `retile`?"""
    _, tile_grid, edges_grid = make_grids
    for ix, iy, iz in itertools.product(*map(xrange, tile_grid.shape)):
        # Ensure pointcloud bounds fall within edges
        bounds = tile_grid[ix, iy, iz].bounds
        for compare, edges, bounds in zip(
                (np.less_equal, np.greater_equal), # both edges inclusive due to outermost edges
                (edges_grid[ix, iy, iz], edges_grid[ix+1, iy+1, iz+1]), # adjacent pointclouds
                (bounds[:3], bounds[3:])): # mins, maxs
            for edge, bound in zip(edges, bounds):
                print compare, edge, bound
                assert compare(edge, bound)
