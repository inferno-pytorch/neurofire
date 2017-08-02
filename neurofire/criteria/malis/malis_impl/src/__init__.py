from _malis_impl import *

# TODO would be nice to have this in python, but I don't want to go to the hassle of correctly
# vectorifying the gt-affinities now
#def constrained_malis_impl(affinities, groundtruth):
#    """
#    Constraine Malis Loss:
#    """
#
#    # compute the groundtruth affinities:
#    gt_affinity = np.zeros_like(affinities, dtype=affinities.dtype)
