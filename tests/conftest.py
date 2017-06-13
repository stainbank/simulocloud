import pytest
import simulocloud.pointcloud
import simulocloud.tiles
import os

def abspath(fname, fdir='data'):
    """Return the absolute filepath of filename in (relative) directory."""
    return os.path.join(os.path.dirname(__file__), fdir, fname)

@pytest.fixture(params=[simulocloud.pointcloud.PointCloud,
                        simulocloud.tiles.Tile])
def pc_las(request, fname='ALS.las'):
    """Set up a pointcloud using single file test data."""
    return request.param.from_las(abspath(fname))
