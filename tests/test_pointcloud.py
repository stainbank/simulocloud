from simulocloud import PointCloud
import json
import numpy as np

""" Test data """

_TEST_XYZ = [[10.0, 12.2, 14.4, 16.6, 18.8],
             [11.1, 13.3, 15.5, 17.7, 19.9],
              [0.1, 2.1, 4.5, 6.7, 8.9]]

_EXPECTED_POINTS = np.array([( 10. ,  11.1,  0.1),
                             ( 12.2,  13.3,  2.1),
                             ( 14.4,  15.5,  4.5),
                             ( 16.6,  17.7,  6.7),
                             ( 18.8,  19.9,  8.9)], 
                            dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])

""" Test functions """

def test_PointCloud_read_directly_from_list():
    """Can PointCloud initialise directly from `[[xs], [ys], [zs]]` ?"""
    assert np.all(PointCloud(_TEST_XYZ).points == _EXPECTED_POINTS)

