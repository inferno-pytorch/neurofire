from __future__ import print_function, division

import numpy as np
import unittest


# hacky includes
import sys
sys.path.append('..')
from bld import malis_impl, constrained_malis_impl

sys.path.append('./pymalis')
import pymalis

import time

class MalisTest(unittest.TestCase):

    def generate_test_data(self, generate_2d=True):
        shape = (100, 100) if generate_2d else (100, 100, 100)
        dim = 2 if generate_2d else 3

        affinities = np.random.random((dim,) + shape).astype('float32')
        groundtruth = np.zeros(shape, dtype='int64')

        current_label = 0
        for x in range(groundtruth.shape[0]):
            for y in range(groundtruth.shape[1]):
                groundtruth[x,y] = current_label
                # change label with probability .3
                if np.random.random() > .7:
                    current_label += 1

        return affinities, groundtruth


    def test_malis_impl(self):
        affinities, groundtruth = self.generate_test_data()
        for pos in (True, False):
            gradients, loss, _, _ = malis_impl(affinities, groundtruth, pos)
            self.assertEqual(gradients.shape, affinities.shape)
            print("Malis test with loss:", loss)


    def test_malis_impl_constrained(self):
        affinities, groundtruth = self.generate_test_data(False)

        # run constrained malis impl
        t_malis_impl = time.time()
        gradients_malis_impl, loss = constrained_malis_impl(affinities, groundtruth)
        gradients_malis_impl *= -1
        gradients_malis_impl / 2.
        print("Runtime for constrained malis_impl:", time.time() - t_malis_impl)
        self.assertEqual(affinities.shape, gradients_malis_impl.shape)

        if True:
            # run constrained malis from pymalis and compare with malis imple output
            t_pymalis = time.time()
            gradients_pos, gradients_neg = pymalis.malis(affinities, groundtruth)
            gradients_pymalis = -(gradients_pos + gradients_neg) / 2.
            print("Runtime for constrained pymalis:", time.time() - t_pymalis)
            self.assertEqual(gradients_pymalis.shape, affinities.shape)

            gradients_pymalis = gradients_pymalis.transpose((1, 2, 3, 0))
            diff = np.isclose(gradients_malis_impl, gradients_pymalis, rtol=1e-2)
            print("Number of equal entries:", np.sum(diff))
            print("                       /", diff.size)
            print("Corresponding to a", np.sum(diff) / diff.size, "% match")
            self.assertTrue(diff.all())


    def test_pymalis(self):
        affinities, groundtruth = self.generate_test_data(False)
        gradients_pos, gradients_neg = pymalis.malis(affinities, groundtruth)
        self.assertEqual(gradients_pos.shape, gradients_neg.shape)


if __name__ == '__main__':
    unittest.main()
