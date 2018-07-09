import torch
import numpy as np
import unittest


class TestArand(unittest.TestCase):
    def build_input(self):
        affinity_image = np.array([[0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [1, 1, 1, 1, 1],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0]]).astype('float32')
        groundtruth_image = np.array([[1, 1, 0, 2, 2],
                                      [1, 1, 0, 2, 2],
                                      [0, 0, 0, 0, 0],
                                      [4, 4, 0, 3, 3],
                                      [4, 4, 0, 3, 3]]).astype('float32')
        affinity_batch = np.array([affinity_image for _ in range(2)])[None, ...]
        groundtruth_batch = groundtruth_image[None, None, ...]
        affinity_batch_tensor = torch.from_numpy(affinity_batch)
        groundtruth_batch_tensor = torch.from_numpy(groundtruth_batch)
        return affinity_batch_tensor, groundtruth_batch_tensor

    def test_arand_error_with_connected_components_on_affinities(self):
        from neurofire.metrics.arand import ArandErrorFromConnectedComponentsOnAffinities
        affinity, groundtruth = self.build_input()
        arand = ArandErrorFromConnectedComponentsOnAffinities(thresholds=[0.5, 0.8],
                                                              invert_affinities=False)
        error = arand(affinity, groundtruth)
        self.assertEqual(error, 0.)


if __name__ == '__main__':
    unittest.main()
