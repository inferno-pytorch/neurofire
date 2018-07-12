import torch
from torch.autograd import Function

try:
    import affogato
    HAVE_AFFOGATO = True
except ImportError:
    HAVE_AFFOGATO = False


class MalisLoss(Function):
    """
    Compute the constrained malis loss.
    """
    def __init__(self, ndim=3):
        import affogato.learning as affl
        assert ndim in (2, 3)
        assert HAVE_AFFOGATO, "Need `affogato` module to compute malis loss"
        self.malis_function = affl.compute_malis_2d if ndim == 2 else affl.compute_malis_3d
        if ndim == 2:
            self.offsets = [[-1, 0], [0, -1]]
        else:
            self.offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

    def forward(self, affinities, segmentation):
        affinities = affinities.numpy()
        segmentation = segmentation.numpy()
        assert affinities.ndim == self.ndim + 1
        assert affinities.shape[1] == self.ndim
        assert affinities.shape[2:] == segmentation.shape[1:]
        loss, gradients = self.malis_function(affinities, segmentation, self.offsets)
        self.save_for_backward(torch.from_numpy(gradients))
        self.seg_shape = segmentation.shape
        # TODO in the old malis code, we return the gradients here, I don't get why
        return loss

    def backward(self):
        gradients, = self.saved_tensors
        target_gradient = gradients.new(*self.seg_shape).zero_()
        return gradients, target_gradient
