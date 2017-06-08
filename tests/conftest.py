import pytest
import simulocloud.pointcloud
import os

def abspath(fname, fdir='data'):
    """Return the absolute filepath of filename in (relative) directory."""
    return os.path.join(os.path.dirname(__file__), fdir, fname)

@pytest.fixture
def pc_las(fname='ALS.las'):
    """Set up a PointCloud instance using single file test data."""
    return simulocloud.pointcloud.PointCloud.from_las(abspath(fname))
