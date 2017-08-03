import unittest
import neurofire.transforms.segmentation as seg
import numpy as np


class TestSegmentation(unittest.TestCase):
    def test_segmentation2affinitiy_random(self):
        # 3D with 3D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 16, 512, 512))
        # Build transform
        transform = seg.Segmentation2Affinities(dim=3)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((16, 512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 3)
        # 3D with 2D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 16, 512, 512))
        # Build transform
        transform = seg.Segmentation2Affinities(dim=2)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((16, 512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 2)
        # 2D with 2D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 512, 512))
        # Build transform
        transform = seg.Segmentation2Affinities(dim=2)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 2)


if __name__ == '__main__':
    unittest.main()