import unittest
import neurofire.transforms.segmentation as seg
import numpy as np


class TestSegmentation(unittest.TestCase):
    def test_segmentation2affinitiy_random(self):
        wannabe_groundtruth = np.random.uniform(size=(1, 16, 512, 512))
        # Build transform
        transform = seg.Segmentation2Affinities()
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((16, 512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 3)


if __name__ == '__main__':
    unittest.main()