from torch.autograd import Function

from malis_impl import malis_impl

# malis loss and cpp impl adapted from:
# https://github.com/naibaf7/caffe/blob/master/include/caffe/layers/malis_loss_layer.hpp
# https://github.com/naibaf7/caffe/blob/master/src/caffe/layers/malis_loss_layer.cpp

# TODO: the caffee implementation parallelizes over the batches in the backward / forward pass
# I have no idea how batches are handled in pytorch, so I haven't done any parallelization yet
# However the gil for 'malis_impl' is lifted, so this CAN be used in multiple threads

# TODO as far as I can tell, this is normal MALIS. I can't see, where in the caffe impl
# constrained MALIS (aka MALA) comes into play

# To create new loss, we inherit from Function:
# https://github.com/pytorch/pytorch/blob/master/torch/autograd/function.py#L123
# see also
# http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
class MalisLoss(Function):
    """
    Malis Loss
    """

    def forward(self, affinities, groundtruth):
        """
        Apply malis forward pass to get the loss.

        Parameters
        ----------
        affinities : torch.Variable wrapping the affinity tensor
        groundtruth : torch.Variable wrapping the groundtruth tensor

        Returns
        -------
        loss: malis loss
        """

        # fist, compute the positive loss and gradients
        pos_gradients, pos_loss, _, _ = malis_impl(
            affinities, groundtruth, True
        )

        # next, compute the negative loss and gradients
        # TODO for constrained malis, we need to somehow restrict the segmentation here
        neg_gradients, neg_loss, _, _ = malis_impl(
            affinities, groundtruth, False
        )

        # save the combined gradient for the backward pass
        self.save_for_backward(
            -(neg_gradients + pos_gradients) / 2.
        )

        # return the combined loss
        return (neg_loss + pos_loss) / 2.

    def backward(self, grad_output):
        """
        Apply malis backward pass to get the gradients.
        """

        gradients, = self.saved_tensors
        return gradients
