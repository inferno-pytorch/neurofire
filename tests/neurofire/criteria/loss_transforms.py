import unittest
import numpy as np
import torch


class TestLossTransforms(unittest.TestCase):
    shape = (50, 50, 50)

    @staticmethod
    def make_segmentation_with_ignore(shape):
        seg = np.zeros(shape, dtype='uint32')
        label = 1
        # make randoms segmentation
        for index in range(seg.size):
            coord = np.unravel_index(index, shape)
            seg[coord] = label
            if np.random.rand() > 0.9:
                label += 1
        # mask 10 % random pixel
        mask = np.random.choice([0, 1],
                                size=seg.size,
                                p=[0.9, 0.1]).reshape(seg.shape).astype('bool')
        seg[mask] = 0
        return seg

    def test_mask_ignore_label(self):
        from neurofire.criteria.loss_transforms import MaskIgnoreLabel
        trafo = MaskIgnoreLabel(0)
        target = self.make_segmentation_with_ignore(self.shape)
        # make dummy torch prediction
        prediction = torch.Tensor(self.shape).uniform_(0, 1)
        masked_prediction, target_ = trafo(prediction, torch.from_numpy(target))
        self.assertTrue(np.allclose(target_.numpy(), target))
        # FIXME this does not really make sense:
        # we can only check if the gradiente were masked properly,
        # after we have computed a loss
        masked_grads = masked_prediction.grad.data.numpy()
        ignore_mask = target == 0
        self.assertTrue((masked_grads[ignore_mask] == 0).all())


if __name__ == '__main__':
    unittest.main()
