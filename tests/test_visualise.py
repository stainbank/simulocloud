"""test_visualise.py
Unit testing for the pointcloud.visualise module
"""
import pytest
from simulocloud.visualise import scatter
from simulocloud.exceptions import BadDims, WrongNDims
from test_pointcloud import pc_las

def test_scatter_rejects_wrong_dims_type(pc_las):
    """Is an error raised when dims argument to scatter is not str?."""
    with pytest.raises(BadDims):
        scatter((pc_las,), pc_las)

def test_scatter_rejects_wrong_dims_length(pc_las):
    """Is an error raised when dims argument to scatter is not str?."""
    with pytest.raises(WrongNDims):
        scatter((pc_las,), 'x')
