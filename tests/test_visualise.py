"""test_visualise.py
Unit testing for the pointcloud.visualise module
"""
import pytest
import simulocloud.visualise
import simulocloud.exceptions

def test_scatter_rejects_wrong_axes_type(pc_las):
    """Is an error raised when axes argument to scatter is not str?."""
    with pytest.raises(simulocloud.exceptions.BadDims):
        simulocloud.visualise.scatter((pc_las,), pc_las)

def test_scatter_rejects_wrong_axes_length(pc_las):
    """Is an error raised when axes argument to scatter is not str?."""
    with pytest.raises(simulocloud.exceptions.WrongNDims):
        simulocloud.visualise.scatter((pc_las,), 'x')
