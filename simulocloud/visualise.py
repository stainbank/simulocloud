""" visualise.py

See, plot and visually explore pointclouds
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import itertools
import simulocloud.exceptions

# Mapping of dimension to index in bounds
_IDIM = {'x': 0, 'y': 1, 'z': 2}

def scatter(pcs, dims, bounds=None, highlight=None, n=10000,
            colours=None, labels=None, title=None, figsize=(6,6)):
    """Create a scatter plot of one or more point clouds in 2D or 3D.
    
    Arguments:
    ----------
    pcs: iterable of PointCloud instances
        point clouds to plot
    dims: str
        dimensions to plot on x, y (and optionally z) axes
        e.g. 'xz' -> x vs z (i.e. 2D cross-section); 'xyz' -> x vs y vs z (3D)
    bounds: pointcloud.Bounds namedtuple (optional)
        (minx, miny, minz, maxx, maxy, mayz) bounds to crop pointclouds to
    highlight: tuple or pointcloud.Bounds nametuple (optional)
        (minx, miny, minz, maxx, maxy, mayz) bounds of area to highlight
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
    # Parse dims as 2D or 3D
    try:
        dims = dims.lower()
        ndims = len(dims)
        projection = {2: None, 3: '3d'}[ndims]
        trace = {2: lambda x0y0x1y1: (_trace_rectangle(*x0y0x1y1),),
                 3: _trace_cuboid}[ndims]
    except(AttributeError):
        raise simulocloud.exceptions.BadDims('dims must be str (not {})'.format(type(dims))) 
    except(KeyError): 
        raise simulocloud.exceptions.WrongNDims('dims must have either 2 or 3 dims (had {})'.format(ndims))
     
    # Set up figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    ax.set_aspect('equal')
    
    # Draw plots
    pcs = _crop_and_sample_pointclouds(pcs, bounds, n)
    for arrs, kwargs in _iter_scatter_args(pcs, dims, colours, labels):
        ax.scatter(s=2, edgecolors='none', *arrs, **kwargs)
   
    # Highlight area
    if highlight is not None:
        hbounds = _reorient_bounds(highlight, dims)
        rects = trace(hbounds)
        for rect in rects:
            ax.plot(*rect, c='fuchsia')
    
    # Annotate figure
    ax.legend()
    ax.set_xlabel(dims[0].upper())
    ax.set_ylabel(dims[1].upper())
    if ndims == 3:
        ax.set_zlabel(dims[2].upper())
    if title is not None:
        ax.set_title(title)
     
    return fig

def _iter_scatter_args(pcs, dims, colours, labels):
    """Yield plotting arrays and matplotlib scatter kwargs per pointcloud."""
    # Generate defaults
    if colours is None:
        colours = _iternones()
    if labels is None:
        # labels = _iteralphabet()
        labels = _iternones()
        
    for pc, colour, label in itertools.izip(pcs, colours, labels):
        arrs = (getattr(pc, dim.lower()) for dim in dims) # extract coordinates
        kwargs = {'c': colour,
                  'label': label}
        yield arrs, kwargs

def _crop_and_sample_pointclouds(pcs, bounds, n):
    """Return generator of cropped point clouds with maximum n points."""
    if bounds is not None:
        pcs = (pc.crop(bounds) for pc in pcs)
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

def _trace_rectangle(x0, y0, x1, y1):
    """Generate 2D coordinates of points outlining rectangle clockwise.
    
    Arguments
    ---------
    x0, y0: float
        coordinates of lowerleftmost rectangle vertex
    x1, y1:  float
        coordinates of upperrightmost rectangle vertex
    
    Returns
    -------
    ndarray (shape: (2, 5))
        x and y coordinates of vertices ABCDA
    
    """
    return np.array([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]).T

def _trace_cuboid(bounds):
    """Generate 3D coordinates of points outlining cuboid faces.
    
    Arguments
    ---------
    bounds: tuple or Bounds namedtuple
        (minx, miny, minz, maxx, maxy, maxz) defining cuboid
    
    Returns
    -------
    ndarray (shape: (6, 3, 5)
        x, y and z coordinates of vertices ABCDA for each cuboid face
    
    """
    cuboid = np.empty((6, 3, 5))
    for ip, ix, iy in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        rect = _trace_rectangle(bounds[ix], bounds[iy], bounds[ix+3], bounds[iy+3])
        for i in (ip, ip+3):
            cuboid[i, ix], cuboid[i, iy] = rect
            cuboid[i, ip] = np.repeat(bounds[i], 5)
    return cuboid

def _reorient_bounds(bounds, dims):
    """Reorder bounds to the specified 2D or 3D spatial orientation.
    
    Arguments
    ---------
    bounds: tuple or bounds namedtuple
        (minx, miny, minz, maxx, maxy, maxz)
        dims: str
        dims to reorient bounds to (e.g. 'yz' or 'xzy')
     
    Returns
    -------
    tuple (len 2*len(dims))
        mins and maxs in order of dims
    
    Usage
    -----
    >>> bounds = Bounds(minx=10., miny=35., minz=6.,
    ...                 maxx=20., maxy=55., maxz=9.)
    >>> _reorient_bounds(bounds, 'xzy') # 3D
    (10.0, 6.0, 35.0, 20.0, 9.0, 55.0)
    >>> _reorient_bounds(bounds, 'zx') # 2D
    (6.0, 10.0, 9.0, 20.0)
        
    """
    return tuple([bounds[_IDIM[dim]+n] for n in (0, 3) for dim in dims])
