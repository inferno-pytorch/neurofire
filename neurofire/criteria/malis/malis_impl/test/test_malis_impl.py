from __future__ import print_function

import numpy as np


import sys
sys.path.append('..')
from bld import malis_impl


def generate_test_data():
    shape = (100, 100)

    affinities = np.random.random(shape + (2,)).astype('float32')
    groundtruth = np.zeros(shape, dtype='uint32')

    current_label = 0
    for x in range(groundtruth.shape[0]):
        for y in range(groundtruth.shape[1]):
            groundtruth[x,y] = current_label
            # change label with probability .3
            if np.random.random() > .7:
                current_label += 1

    return affinities, groundtruth


def test_malis_impl_(pos):
    affinities, groundtruth = generate_test_data()
    gradients, loss, _, _ = malis_impl(affinities, groundtruth, pos)
    assert gradients.shape == affinities.shape
    print("Test passed for pos =", pos, "with loss = ", loss)


if __name__ == '__main__':
    test_malis_impl_(False)
    test_malis_impl_(True)
