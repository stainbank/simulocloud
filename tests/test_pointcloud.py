from simulocloud import PointCloud, Bounds
from simulocloud.exceptions import EmptyPointCloud
import pytest
import numpy as np
import cPickle as pkl
import os

""" Test data """
# Data type used for arrays
_DTYPE = np.float64
_INPUT_DATA = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
               [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]]

@pytest.fixture
def expected_DATA_points():
    """The points array that should be generated from `input_list`."""
    return np.array([( 0. ,  1. ,  2. ),
                     ( 0.1,  1.1,  2.1),
                     ( 0.2,  1.2,  2.2),
                     ( 0.3,  1.3,  2.3),
                     ( 0.4,  1.4,  2.4),
                     ( 0.5,  1.5,  2.5),
                     ( 0.6,  1.6,  2.6),
                     ( 0.7,  1.7,  2.7),
                     ( 0.8,  1.8,  2.8),
                     ( 0.9,  1.9,  2.9)],
                    dtype=[('x', _DTYPE), ('y', _DTYPE), ('z', _DTYPE)])

@pytest.fixture
def input_array():
    """Simple [xs, ys, zs] numpy.ndarray of `input_list`."""
    return np.array(_INPUT_DATA, dtype=_DTYPE)

@pytest.fixture
def expected_las_points(fname='ALS_points.pkl'):
    """The points array that should be generated from the example las data."""
    with open(abspath(fname), 'rb') as o:
        points = pkl.load(o)
    return points

@pytest.fixture
def pc(input_array):
    """Set up a `PointCloud` instance using test data."""
    return PointCloud(input_array)

@pytest.fixture
def none_bounds():
    """A Bounds nametuple with all bounds set to None."""
    return Bounds(*(None,)*6)

@pytest.fixture
def inf_bounds():
    """A Bounds namedtuple with all bounds set to inf/-inf."""
    return Bounds(*(np.inf,)*3 + (np.inf,)*3)
""" Helper functions """

def abspath(fname, fdir='data'):
    """Return the absolute filepath of filename in (relative) directory."""
    return os.path.join(os.path.dirname(__file__), fdir, fname)

""" Test functions """

def test_PointCloud_read_from_array(input_array, expected_DATA_points):
    """Can PointCloud initialise directly from a `[xs, ys, zs]` array?"""
    assert np.all(PointCloud(input_array).points == expected_DATA_points)

def test_PointCloud_read_from_las(expected_las_points, fname='ALS.las'):
    """Can PointCloud be constructed from a `.las` file?"""
    assert np.all(PointCloud.from_las(abspath(fname)).points == expected_las_points)

def test_arr_generation(pc, input_array):
    """Does PointCloud.arr work as expected?."""
    assert np.all(pc.arr == input_array.T)

def test_bounds_returns_accurate_boundary_box(pc):
    """Does PointCloud.bounds accurately describe the bounding box?"""
    assert pc.bounds == tuple((f(c) for f in (min, max) for c in _INPUT_DATA)) 

def test_len_works(pc):
    """Does __len__() report the correct number of points?"""
    # Assumes lists in _INPUT_DATA are consistent length
    assert len(pc) == len(_INPUT_DATA[0])

def test_cropping_with_none_bounds(pc, none_bounds):
    """Does no PointCloud cropping occur when bounds of None are used?"""
    assert np.all(pc.crop(none_bounds).points == pc.points)

@pytest.mark.parametrize('c', ('x', 'y', 'z'))
def test_cropping_is_lower_bounds_inclusive(pc, none_bounds, c):
    """Does PointCloud cropping preserve values at lower bounds?"""
    # Ensure a unique point used as minimum bound
    sorted_points = np.sort(pc.points, order=[c])
    for i, minc in enumerate(sorted_points[c]):
        if i < 1: continue # at least one point must be out of bounds
        if minc != sorted_points[i-1][c]: 
            lowest_point = sorted_points[i]
            break
    
    # Apply lower bound cropping to a single dimension
    pc = pc.crop(none_bounds._replace(**{'min'+c: minc}))

    assert np.sort(pc.points, order=c)[0] == lowest_point

@pytest.mark.parametrize('c', ('x', 'y', 'z'))
def test_cropping_is_upper_bounds_exclusive(pc, none_bounds, c):
    """Does PointCloud cropping omit values at upper bounds?"""
    # Ensure a unique point used as maximum bound
    rev_sorted_points = np.sort(pc.points, order=[c])[::-1]
    for i, maxc in enumerate(rev_sorted_points[c]):
        if maxc != rev_sorted_points[i+1][c]:
            oob_point = rev_sorted_points[i]
            highest_point = rev_sorted_points[i+1]
            break

    # Apply upper bound cropping to a single dimension
    pc = pc.crop(none_bounds._replace(**{'max'+c: maxc}))

    assert np.sort(pc.points, order=c)[-1] == highest_point and oob_point not in pc.points

def test_cropping_to_nothing_raises_exception_when_specified(pc, inf_bounds):
    """Does PointCloud cropping refuse to return an empty PointCloud?"""
    with pytest.raises(EmptyPointCloud):
        pc.crop(inf_bounds, return_empty=False)

def test_cropping_to_nothing_returns_empty(pc, inf_bounds):
    """Does PointCloud cropping return an empty PointCloud when asked?"""
    assert not len(pc.crop(inf_bounds, return_empty=True))
