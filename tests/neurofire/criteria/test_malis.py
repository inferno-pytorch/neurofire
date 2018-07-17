import unittest
import numpy as np
import torch
from torch.autograd import Variable

try:
    from base_test import BaseTest
except ImportError:
    from .base_test import BaseTest


class TestMalis(BaseTest):

    def test_malis_3d(self):
        from neurofire.criteria.malis import MalisLoss
        seg = torch.from_numpy(self.make_segmentation(self.shape)[None, None])
        aff_shape = (1, 3) + self.shape

        pred = torch.rand(*aff_shape, dtype=torch.float, requires_grad=True)
        # prediction = Variable(torch.Tensor(*pshape).uniform_(0, 1), requires_grad=True)
        criterion = MalisLoss(ndim=3)

        loss = criterion.forward(pred, seg)
        # FIXME why do we have to set this manually ?
        loss.requires_grad = True
        self.assertNotEqual(loss.item(), 0)
        loss.backward()

        # FIXME no gradients although we call the backward pass
        print(pred.grad)
        grads = pred.grad.data.numpy().squeeze()
        self.assertEqual(grads.shape, pred.shape)
        self.assertFalse(np.allclose(grads, 0))


if __name__ == '__main__':
    unittest.main()
