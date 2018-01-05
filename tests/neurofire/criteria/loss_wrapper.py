import unittest
import numpy as np
import torch
from torch.autograd import Variable

# FIXME somehow, i can't get this to work
# from torch.nn.modules.loss import CrossEntropyLoss

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from inferno.io.transform.base import Compose
from base_test import BaseTest


class TestLossWrapper(BaseTest):

    def test_loss_wrapper_affinity_masking(self):
        from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel
        from neurofire.criteria.loss_transforms import RemoveSegmentationFromTarget
        from neurofire.criteria.loss_wrapper import LossWrapper
        from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets

        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (9, 0, 0), (0, 9, 0), (0, 0, 9),
                   (9, 4, 0), (4, 9, 0), (9, 0, 9)]

        trafos = Compose(MaskTransitionToIgnoreLabel(offsets, ignore_label=0),
                         RemoveSegmentationFromTarget())

        aff_trafo = Segmentation2AffinitiesFromOffsets(3, offsets, retain_segmentation=True,
                                                       add_singleton_channel_dimension=True)

        seg = self.make_segmentation_with_ignore(self.shape)
        ignore_mask = self.brute_force_transition_masking(seg, offsets)
        target = Variable(torch.Tensor(aff_trafo(seg.astype('float32'))[None]),
                          requires_grad=False)

        tshape = target.size()
        pshape = (tshape[0], tshape[1] - 1) + tshape[2:]
        prediction = Variable(torch.Tensor(*pshape).uniform_(0, 1), requires_grad=True)

        # apply cross entropy loss
        # FIXME error with Cross Entropy
        # criterion = CrossEntropyLoss()
        criterion = SorensenDiceLoss()
        wrapper = LossWrapper(criterion, trafos)
        loss = wrapper.forward(prediction, target)
        loss.backward()

        grads = prediction.grad.data.numpy().squeeze()
        self.assertEqual(grads.shape, ignore_mask.shape)
        self.assertTrue((grads[ignore_mask] == 0).all())
        self.assertFalse(np.sum(grads[np.logical_not(ignore_mask)]) == 0)


if __name__ == '__main__':
    unittest.main()
