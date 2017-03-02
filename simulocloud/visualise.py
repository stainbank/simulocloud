""" visualise.py

See, plot and visually explore pointclouds
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import izip
from .exceptions import BadAxes, InvalidAxesDims

def scatter(pcs, axes, bounds=None, n=10000, colours=None, labels=None,
            title=None, figsize=(6,6)):
    """Create a scatter plot of one or more point clouds in 2D or 3D.
    
    Arguments:
    ----------
    pcs: iterable of PointCloud instances
        point clouds to plot
    axes: str
        dimensions to plot on x, y (and optionally z) axes
        e.g. 'xz' -> x vs z (i.e. 2D cross-section); 'xyz' -> x vs y vs z (3D)
    bounds: pointcloud.Bounds namedtuple (optional)
        (minx, miny, minz, maxx, maxy, mayz) bounds to crop pointclouds to
    n: int (default: 1e4)
        max number of points to plot per point cloud
    colours: iterable of valid matplotlib color arguments (optional)
        colours to use for each pointcloud
    labels: iterable of str (optional)
        labels for each pointcloud
        a, b, c etc. used by default
    title: str (optional)
        figure title
    figsize: tuple (default: (6,6)
        (width, height) figure dimensions in inches

    Returns:
    --------
    matplotlib.figure.Figure instance
    
    """
    # Parse axes as 2D or 3D
    try:
        axes = axes.lower()
        ndims = len(axes)
        projection = {2: None, 3: '3d'}[ndims]
    except(AttributeError):
        raise BadAxes('axes must be str (not {})'.format(type(axes))) 
    except(KeyError): 
        raise InvalidAxesDims('axes must have either 2 or 3 dims (had {})'.format(ndims))
     
    # Set up figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    ax.set_aspect('equal')
    
    # Draw plots
    pcs = _crop_and_sample_pointclouds(pcs, bounds, n)
    for arrs, kwargs in _iter_scatter_args(pcs, axes, colours, labels):
        ax.scatter(s=2, edgecolors='none', *arrs, **kwargs)
    
    # Annotate figure
    ax.legend()
    ax.set_xlabel(axes[0].upper())
    ax.set_ylabel(axes[1].upper())
    if ndims == 3:
        ax.set_zlabel(axes[2].upper())
    if title is not None:
        ax.set_title(title)
     
    return fig

def _iter_scatter_args(pcs, axes, colours, labels):
    """Yield plotting arrays and matplotlib scatter kwargs per pointcloud."""
    # Generate defaults
    if colours is None:
        colours = _iternones()
    if labels is None:
        labels = _iteralphabet()
        
    for pc, colour, label in izip(pcs, colours, labels):
        arrs = (pc.points[dim.lower()] for dim in axes) # extract coordinates
        kwargs = {'c': colour,
                  'label': label}
        yield arrs, kwargs

def _crop_and_sample_pointclouds(pcs, bounds, n):
    """Return generator of cropped point clouds with maximum n points."""
    if bounds is not None:
        pcs = (pc.crop(*bounds) for pc in pcs)
    return (pc.downsample(n) if len(pc)>n else pc for pc in pcs) 

def _iternones():
    """Return infinite generator yielding None."""
    while True:
        yield None

def _iteralphabet():
    """Return infinite generator yielding str in sequence a..z, aa..zz, etc."""
    n = 1
    while True:  
        for i in xrange(26):
            yield chr(97+i)*n
        else:
            n += 1 
