from simulocloud import PointCloud
import pytest
import numpy as np
import cPickle as pkl
import os

""" Test data """

@pytest.fixture
def input_list():
    """Simple [xs, ys, zs] coordinates."""
    return [[10.0, 12.2, 14.4, 16.6, 18.8],
            [11.1, 13.3, 15.5, 17.7, 19.9],
            [0.1, 2.1, 4.5, 6.7, 8.9]]

@pytest.fixture
def expected_list_points():
    """The points array that should be generated from `input_list`."""
    return np.array([( 10. ,  11.1,  0.1),
                     ( 12.2,  13.3,  2.1),
                     ( 14.4,  15.5,  4.5),
                     ( 16.6,  17.7,  6.7),
                     ( 18.8,  19.9,  8.9)], 
                    dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])

@pytest.fixture
def expected_las_points(fname='ALS_points.pkl'):
    """The points array that should be generated from the example las data."""
    with open(abspath(fname), 'rb') as o:
        points = pkl.load(o)
    return points

""" Helper functions """

def abspath(fname, fdir='data'):
    """Return the absolute filepath of filename in (relative) directory."""
    return os.path.join(os.path.dirname(__file__), fdir, fname)

""" Test functions """

def test_PointCloud_read_directly_from_list(input_list, expected_list_points):
    """Can PointCloud initialise directly from `[xs, ys, zs]` ?"""
    assert np.all(PointCloud(input_list).points == expected_list_points)

def test_PointCloud_read_from_las(expected_las_points, fname='ALS.las'):
    """Can PointCloud be constructed from a `.las` file?"""
    assert np.all(PointCloud.from_las(abspath(fname)).points == expected_las_points)
