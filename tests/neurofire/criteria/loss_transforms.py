import unittest
import numpy as np
import torch
from torch.autograd import Variable

# FIXME somehow, i can't get this to work
# from torch.nn.modules.loss import CrossEntropyLoss

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss


class TestLossTransforms(unittest.TestCase):
    shape = (1, 1, 50, 50, 50)

    @staticmethod
    def make_segmentation_with_ignore(shape):
        seg = np.zeros(shape, dtype='int32')
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
        from neurofire.transforms.segmentation import Segmentation2Membranes
        trafo = MaskIgnoreLabel(0)
        seg_trafo = Segmentation2Membranes()

        seg = self.make_segmentation_with_ignore(self.shape)
        ignore_mask = seg == 0
        target = seg_trafo(seg)

        # make dummy torch prediction
        prediction = Variable(torch.Tensor(*self.shape).uniform_(0, 1), requires_grad=True)
        masked_prediction, _ = trafo(prediction, seg_var)
        self.assertEqual(prediction.size(), masked_prediction.size())
        self.assertEqual(target.size(), seg_var.size())
        self.assertTrue(np.allclose(target.data.numpy(), target))

        # apply a loss to the prediction and check that the
        # masked parts are actually zero

        # binarize the targets with mod2
        positive_mask = target % 2 == 0
        target[positive_mask] = 1
        target[np.logical_not(positive_mask)] = 0
        # target_var = Variable(torch.from_numpy(target), requires_grad=False).long()
        target_var = Variable(torch.from_numpy(target), requires_grad=False).float()

        # apply cross entropy loss
        # FIXME error with Cross Entropy
        # criterium = CrossEntropyLoss()
        criterium = SorensenDiceLoss()
        loss = criterium(masked_prediction, target_var)
        loss.backward()

        # FIXME why no gradients ?
        grads = masked_prediction.grad.data.numpy()
        self.assertTrue((grads[ignore_mask] == 0).all())
        self.assertTrue((grads[np.logical_not(ignore_mask)] != 0).all())

    def test_transition_mask(self):
        from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel
        from neurofire.transforms.segmentation import Segmentation2AffinitiesFromOffsets

        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (9, 0, 0), (0, 9, 0), (0, 0, 9),
                   (9, 4, 0), (4, 9, 0), (9, 0, 9)]
        trafo = MaskTransitionToIgnoreLabel(offsets, ignore_label=0)
        aff_trafo = Segmentation2AffinitiesFromOffsets(3, offsets, retain_segmentation=True)

        seg = self.make_segmentation_with_ignore(self.shape)
        target = aff_trafo(seg)
        ignore_mask = seg == 0


if __name__ == '__main__':
    unittest.main()
