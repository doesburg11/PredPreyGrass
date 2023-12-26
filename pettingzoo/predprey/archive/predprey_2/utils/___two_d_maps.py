import numpy as np
from scipy.ndimage import zoom


def rectangle_map(xs, ys):
    """Returns a 2D 'map' 

    Map is a 2D numpy array
    xb and yb are buffers for each dim representing the ratio of the map to leave open on each side
    """
    rmap = np.zeros((xs, ys), dtype=np.int32)
    return rmap



