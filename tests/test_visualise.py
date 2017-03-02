"""test_visualise.py
Unit testing for the pointcloud.visualise module
"""
import pytest
from simulocloud.visualise import scatter
from simulocloud.exceptions import BadAxes, InvalidAxesDims
from test_pointcloud import pc_las

def test_scatter_rejects_wrong_axes_type(pc_las):
    """Is an error raised when axes argument to scatter is not str?."""
    with pytest.raises(BadAxes):
        scatter((pc_las,), pc_las)

def test_scatter_rejects_wrong_axes_length(pc_las):
    """Is an error raised when axes argument to scatter is not str?."""
    with pytest.raises(InvalidAxesDims):
        scatter((pc_las,), 'x')
