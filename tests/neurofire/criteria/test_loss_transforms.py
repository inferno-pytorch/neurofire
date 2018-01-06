import unittest
import numpy as np
import torch
from torch.autograd import Variable

# FIXME somehow, i can't get this to work
# from torch.nn.modules.loss import CrossEntropyLoss

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from .base_test import BaseTest


class TestLossTransforms(BaseTest):

    def test_mask_ignore_label(self):
        from neurofire.criteria.loss_transforms import MaskIgnoreLabel
        from neurofire.transform.segmentation import Segmentation2Membranes
        trafo = MaskIgnoreLabel(0)
        seg_trafo = Segmentation2Membranes()

        seg = self.make_segmentation_with_ignore(self.shape)
        ignore_mask = seg == 0
        seg_var = Variable(torch.from_numpy(seg.astype('float32')))

        # make membrane target
        target = torch.from_numpy(seg_trafo(seg))
        target = Variable(target, requires_grad=False)
        self.assertEqual(target.size(), seg_var.size())

        # make dummy torch prediction
        prediction = Variable(torch.Tensor(*self.shape).uniform_(0, 1), requires_grad=True)
        # NOTE Can't check that target is not altered here, because we give the segmentation and not
        # the target to the transformation
        # We could check the segmentation, but in the long run, we wan't something like
        # retain_segmentation also for `Segmentation2Membranes`
        masked_prediction, _ = trafo(prediction, seg_var)

        self.assertEqual(prediction.size(), masked_prediction.size())

        # apply a loss to the prediction and check that the
        # masked parts are actually zero

        # apply cross entropy loss
        # FIXME error with Cross Entropy
        # criterium = CrossEntropyLoss()
        criterium = SorensenDiceLoss()
        loss = criterium(masked_prediction, target)
        loss.backward()

        grads = prediction.grad.data.numpy()
        self.assertTrue((grads[ignore_mask] == 0).all())
        self.assertTrue((grads[np.logical_not(ignore_mask)] != 0).all())

    def test_transition_mask(self):
        from neurofire.criteria.loss_transform import MaskTransitionToIgnoreLabel
        from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets

        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (9, 0, 0), (0, 9, 0), (0, 0, 9),
                   (9, 4, 0), (4, 9, 0), (9, 0, 9)]
        trafo = MaskTransitionToIgnoreLabel(offsets, ignore_label=0)
        aff_trafo = Segmentation2AffinitiesFromOffsets(3, offsets, retain_segmentation=True,
                                                       add_singleton_channel_dimension=True)

        seg = self.make_segmentation_with_ignore(self.shape)
        ignore_mask = self.brute_force_transition_masking(seg, offsets)
        target = Variable(torch.Tensor(aff_trafo(seg.astype('float32'))[None]),
                          requires_grad=False)

        tshape = target.size()
        pshape = (tshape[0], tshape[1] - 1) + tshape[2:]
        prediction = Variable(torch.Tensor(*pshape).uniform_(0, 1), requires_grad=True)
        masked_prediction, target_ = trafo(prediction, target)

        self.assertEqual(masked_prediction.size(), prediction.size())
        self.assertEqual(target.size(), target_.size())
        self.assertTrue(np.allclose(target.data.numpy(), target_.data.numpy()))

        # apply cross entropy loss
        # FIXME error with Cross Entropy
        # criterium = CrossEntropyLoss()
        criterium = SorensenDiceLoss()
        target = target[:, 1:]
        self.assertEqual(target.size(), masked_prediction.size())
        loss = criterium(masked_prediction, target)
        loss.backward()

        grads = prediction.grad.data.numpy().squeeze()
        self.assertEqual(grads.shape, ignore_mask.shape)
        self.assertTrue((grads[ignore_mask] == 0).all())
        self.assertFalse(np.sum(grads[np.logical_not(ignore_mask)]) == 0)


if __name__ == '__main__':
    unittest.main()
