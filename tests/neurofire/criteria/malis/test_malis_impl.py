from __future__ import print_function, division

import time
import numpy as np
import unittest

from neurofire.criteria.malis.malis import malis_impl, constrained_malis_impl

def init_pymalis():
    from subprocess import call
    call(['git', 'submodule', 'update', '--init'])

# Jan Funke's pymalis implementation for refernce from
# https://github.com/funkey/pymalis
import sys
sys.path.append('./pymalis')
try:
    import pymalis
except ImportError:
    init_pymalis()
    import pymalis

class TestMalisImpl(unittest.TestCase):

    def generate_test_data(self, generate_2d=True):
        shape = (100, 100) if generate_2d else (100, 100, 100)
        dim = 2 if generate_2d else 3

        affinities = np.random.random((dim,) + shape).astype('float32')
        aff_min, aff_max = affinities.min(), affinities.max()
        assert aff_min >= 0.
        assert aff_max <= 1.

        groundtruth = np.zeros(shape, dtype='int64')

        current_label = 0
        for x in range(groundtruth.shape[0]):
            for y in range(groundtruth.shape[1]):
                groundtruth[x,y] = current_label
                # change label with probability .3
                if np.random.random() > .7:
                    current_label += 1

        return affinities, groundtruth


    def generate_toy_data(self):
        seg = np.zeros((1, 5, 5), dtype='int64')
        seg[:,:,:2] = 1
        seg[:,:,2:] = 2

        aff = np.ones((3, 1, 5, 5), dtype='float32')
        aff[2, :, :2, 2] = 0
        aff[2, :, 3:, 2] = 0
        aff[2, :, 2, 2] = 0.8

        return aff, seg

    #
    # tests on toy data
    #

    def test_malis_impl_toy(self):
        aff, seg = self.generate_toy_data()
        gradients = constrained_malis_impl(aff, seg)

        self.assertEqual(np.sum(gradients == 0), gradients.size - 1)
        self.assertNotEqual(gradients[2, 0, 2, 2], 0.)
        self.assertAlmostEqual(gradients[2, 0, 2, 2], -aff[2, 0, 2, 2])

    def test_pymalis_toy(self):
        aff, seg = self.generate_toy_data()
        gradients_pos, gradients_neg = pymalis.malis(aff, seg)

        gradients = gradients_pos = gradients_neg

        self.assertEqual(np.sum(gradients == 0), gradients.size - 1)
        self.assertNotEqual(gradients[2, 0, 2, 2], 0.)
        self.assertAlmostEqual(gradients[2, 0, 2, 2], -aff[2, 0, 2, 2])

    #
    # simple random tests only check that shapes agree
    #

    def test_malis_impl_simple(self):
        for use_2d in (True, False):
            affinities, groundtruth = self.generate_test_data(use_2d)
            for pos in (True, False):
                gradients, loss, _, _ = malis_impl(affinities, groundtruth, pos)
                self.assertEqual(gradients.shape, affinities.shape)


    def test_pymalis_simple(self):
        affinities, groundtruth = self.generate_test_data(False)
        gradients_pos, gradients_neg = pymalis.malis(affinities, groundtruth)
        self.assertEqual(gradients_pos.shape, gradients_neg.shape)


    #
    # test that results of malis_impl and pymalis on random data agree
    #

    # this only works with custom pymalis
    def _test_malis_impl_components(self):
        affinities, groundtruth = self.generate_test_data(False)
        for pos in (True, False):
            gradients_malis_impl, _, _, _ = malis_impl(affinities, groundtruth, pos)
            gradients_pymalis = pymalis.simple_malis(affinities, groundtruth, pos)
            self.assertEqual(gradients_pymalis.shape, gradients_malis_impl.shape)

            diff = np.isclose(gradients_malis_impl, gradients_pymalis, rtol=1e-2)
            print("Comparison for pos =", pos)
            print("Number of equal entries:", np.sum(diff))
            print("                       /", diff.size)
            print("Corresponding to a", np.sum(diff) / diff.size * 100, "% match")
            self.assertTrue(diff.all())


    def test_malis_impl_constrained(self):
        affinities, groundtruth = self.generate_test_data(False)

        # run constrained malis impl
        t_malis_impl = time.time()
        gradients_malis_impl = constrained_malis_impl(affinities, groundtruth)
        print("Runtime for constrained malis_impl:", time.time() - t_malis_impl)
        self.assertEqual(affinities.shape, gradients_malis_impl.shape)

        # run constrained malis from pymalis and compare with malis imple output
        t_pymalis = time.time()
        gradients_pos, gradients_neg = pymalis.malis(affinities, groundtruth)
        gradients_pymalis = gradients_pos + gradients_neg
        print("Runtime for constrained pymalis:", time.time() - t_pymalis)
        self.assertEqual(gradients_pymalis.shape, affinities.shape)

        diff = np.isclose(gradients_malis_impl, gradients_pymalis, rtol=1e-3)
        print("Number of equal entries:", np.sum(diff))
        print("                       /", diff.size)
        print("Corresponding to a", np.sum(diff) / diff.size * 100, "% match")
        self.assertTrue(diff.all())



if __name__ == '__main__':
    unittest.main()
