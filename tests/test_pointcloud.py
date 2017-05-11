from simulocloud import PointCloud, Bounds
from simulocloud.exceptions import EmptyPointCloud
from laspy.file import File
import pytest
import numpy as np
import cPickle as pkl
import os

""" Constants and fixtures """
# Data type used for arrays
_DTYPE = np.float64
_INPUT_DATA = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
               [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]]

@pytest.fixture
def input_array():
    """Simple [xs, ys, zs] numpy.ndarray of input_list."""
    return np.array(_INPUT_DATA, dtype=_DTYPE)

@pytest.fixture
def expected_las_arr(fname='ALS.las'):
    """The array of points held in the example las data"""
    with File(abspath(fname)) as f:
        return np.array([f.x, f.y, f.z]).T

@pytest.fixture
def pc_arr(input_array):
    """Set up a PointCloud instance using array test data."""
    return PointCloud(input_array)

@pytest.fixture
def pc_arr_x10(pc_arr):
    """Multiply pc_arr values by 10."""
    return PointCloud((pc_arr.arr*10).T)

@pytest.fixture
def pc_las(fname='ALS.las'):
    """Set up a PointCloud instance using single file test data."""
    return PointCloud.from_las(abspath(fname))

@pytest.fixture
def pc_multilas(fdir='ALS_tiles'):
    """Set up a PointCloud instance using multiple file test data, tiled from pc_las."""
    fpaths = [abspath(fname, os.path.join('data', fdir)) for fname in os.listdir(abspath(fdir))]
    return PointCloud.from_las(*fpaths)

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

def test_PointCloud_read_from_array(pc_arr, input_array):
    """Can PointCloud initialise directly from a [xs, ys, zs] array?"""
    assert np.allclose(pc_arr.arr, input_array.T)

def test_PointCloud_read_from_single_las(pc_las, expected_las_arr):
    """Can PointCloud be constructed from a single .las file?"""
    assert np.allclose(pc_las.arr, expected_las_arr)

def test_PointCloud_from_multiple_las(pc_multilas, pc_las):
    """Can PointCloud be constructed from multiple .las files?"""
    assert len(pc_multilas) == len(pc_las)

def test_PointCloud_from_tiles():
    """Can a specific-area PointCloud be constructed from multiple .las files?"""
    assert False # Haven't written test yet

def test_empty_PointCloud():
    """Is the PointCloud generated from `None` empty?"""
    assert not len(PointCloud(None))

def test_arr_generation(pc_arr, input_array):
    """Does PointCloud.arr work as expected?."""
    assert np.allclose(pc_arr.arr, input_array.T)

def test_bounds_returns_accurate_boundary_box(pc_arr):
    """Does PointCloud.bounds accurately describe the bounding box?"""
    assert pc_arr.bounds == tuple((f(c) for f in (min, max) for c in _INPUT_DATA)) 

def test_len_works(pc_arr):
    """Does __len__() report the correct number of points?"""
    # Assumes lists in _INPUT_DATA are consistent length
    assert len(pc_arr) == len(_INPUT_DATA[0])

def test_PointCloud_addition_len(pc_arr):
    """Does PointCloud addition combine the arrays?"""
    pc = pc_arr + pc_arr
    assert len(pc) == len(pc_arr)*2

def test_PointCloud_addition_values(pc_arr, pc_arr_x10):
    """Does PointCloud addition combine the values appropriately?"""
    pc = pc_arr + pc_arr_x10
    # Mins from small, maxs from big
    assert (pc.bounds[:3] == pc_arr.bounds[:3]) and (pc.bounds[3:] == pc_arr_x10.bounds[3:])

@pytest.mark.parametrize('i,dim', [(0, 'x'), (1, 'y'), (2,'z')])
def test_dim_attributes_are_accurate(input_array, pc_arr, i, dim):
    """Do the x, y and z attributes retrieve the dimension array correctly?"""
    assert np.allclose(getattr(pc_arr, dim), input_array[i])

def test_cropping_with_none_bounds(pc_arr, none_bounds):
    """Does no PointCloud cropping occur when bounds of None are used?"""
    assert np.allclose(pc_arr.crop(*none_bounds).arr, pc_arr.arr)

@pytest.mark.parametrize('dim', ('x', 'y', 'z'))
def test_cropping_is_lower_bounds_inclusive(pc_arr, none_bounds, dim):
    """Does PointCloud cropping preserve values at lower bounds?"""
    # Ensure a unique point used as minimum bound
    sorted_points = np.sort(pc_arr.points, order=[dim])
    for i, mindim in enumerate(sorted_points[dim]):
        if i < 1: continue # at least one point must be out of bounds
        if mindim != sorted_points[i-1][dim]: 
            lowest_point = sorted_points[i]
            break
    
    # Apply lower bound cropping to a single dimension
    pc_cropped = pc_arr.crop(**{'min'+dim: mindim})
    
    assert np.sort(pc_cropped.points, order=dim)[0] == lowest_point

@pytest.mark.parametrize('c', ('x', 'y', 'z'))
def test_cropping_is_upper_bounds_exclusive(pc_arr, none_bounds, c):
    """Does PointCloud cropping omit values at upper bounds?"""
    # Ensure a unique point used as maximum bound
    rev_sorted_points = np.sort(pc_arr.points, order=[c])[::-1]
    for i, maxc in enumerate(rev_sorted_points[c]):
        if maxc != rev_sorted_points[i+1][c]:
            oob_point = rev_sorted_points[i]
            highest_point = rev_sorted_points[i+1]
            break
    # Apply upper bound cropping to a single dimension
    pc_arr = pc_arr.crop(**{'max'+c: maxc})
    
    assert (np.sort(pc_arr.points, order=c)[-1] == highest_point) and (
           oob_point not in pc_arr.points)

def test_cropping_to_nothing_raises_exception_when_specified(pc_arr, inf_bounds):
    """Does PointCloud cropping refuse to return an empty PointCloud?"""
    with pytest.raises(EmptyPointCloud):
        pc_arr.crop(*inf_bounds, return_empty=False)

def test_cropping_to_nothing_returns_empty(pc_arr, inf_bounds):
    """Does PointCloud cropping return an empty PointCloud when asked?"""
    assert not len(pc_arr.crop(*inf_bounds, return_empty=True))

def test_PointCloud_exports_transparently_to_txt(pc_arr, tmpdir):
    """Is the file output by PointCloud.to_txt identical to the input?"""
    fpath = tmpdir.join("_INPUT_DATA.txt").strpath
    pc_arr.to_txt(fpath) 

    assert np.allclose(pc_arr.arr, PointCloud(np.loadtxt(fpath)).arr)

def test_PointCloud_exports_transparently_to_las(pc_las, tmpdir):
    """Are the points in the file output by PointCloud.to_las identical to input?"""
    fpath = tmpdir.join('pc_las.las').strpath
    pc_las.to_las(fpath)
    
    assert np.allclose(pc_las.arr, PointCloud.from_las(fpath).arr)

def test_PointCloud_can_downsample(pc_las):
    """Does downsampling a pointcloud to len n preserve n points?"""
    n = int(len(pc_las)/10) # decimate pointcloud
    pc = pc_las.downsample(n)
    assert len(pc) == n and len(np.intersect1d(pc_las.points, np.unique(pc.points))) == len(pc.points)
