import unittest
import numpy as np
from neurofire.criteria.malis.malis import MalisLoss, ConstrainedMalisLoss, Malis
import torch
from torch.autograd import Variable


class TestMalis(unittest.TestCase):
    BATCH_SIZE = 10

    def generate_test_data(self, generate_2d=True):
        shape = (100, 100) if generate_2d else (100, 100, 100)
        dim = 2 if generate_2d else 3

        affinities = np.random.random((dim,) + shape).astype('float32')
        groundtruth = np.zeros(shape, dtype='int64')

        current_label = 0
        for x in range(groundtruth.shape[0]):
            for y in range(groundtruth.shape[1]):
                groundtruth[x, y] = current_label
                # change label with probability .3
                if np.random.random() > .7:
                    current_label += 1

        return affinities, groundtruth

    def test_malis_loss(self):
        affinities, ground_truth = self.generate_test_data()
        affinities = np.array([affinities.copy() for _ in range(self.BATCH_SIZE)])
        ground_truth = np.array([ground_truth.copy() for _ in range(self.BATCH_SIZE)])
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        criterion = MalisLoss()
        # Evaluate criterion
        loss = criterion(affinities, ground_truth).sum()
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)

    def test_constrained_malis_loss(self):
        affinities, ground_truth = self.generate_test_data()
        affinities = np.array([affinities.copy() for _ in range(self.BATCH_SIZE)])
        ground_truth = np.array([ground_truth.copy() for _ in range(self.BATCH_SIZE)])
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        criterion = ConstrainedMalisLoss()
        # Evaluate criterion
        loss = criterion(affinities, ground_truth).sum()
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)

    def test_malis(self):
        affinities, ground_truth = self.generate_test_data()
        affinities = np.array([affinities.copy() for _ in range(self.BATCH_SIZE)])
        ground_truth = np.array([ground_truth.copy() for _ in range(self.BATCH_SIZE)])
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        malis = Malis(constrained=True)
        # Forward and back
        loss = malis(affinities, ground_truth)
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)


if __name__ == '__main__':
    unittest.main()
