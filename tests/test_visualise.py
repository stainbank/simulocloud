"""test_visualise.py
Unit testing for the pointcloud.visualise module
"""
import pytest
import simulocloud.visualise
import simulocloud.exceptions
from test_pointcloud import pc_las # need to put shared fixtures in conftest.py

def test_scatter_rejects_wrong_dims_type(pc_las):
    """Is an error raised when dims argument to scatter is not str?."""
    with pytest.raises(simulocloud.exceptions.BadDims):
        simulocloud.visualise.scatter((pc_las,), pc_las)

def test_scatter_rejects_wrong_dims_length(pc_las):
    """Is an error raised when dims argument to scatter is not str?."""
    with pytest.raises(simulocloud.exceptions.WrongNDims):
        simulocloud.visualise.scatter((pc_las,), 'x')
