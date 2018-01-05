import unittest
import neurofire.transform.volume as vol
import numpy as np
# import torch


# TODO proper tests on random data
class TestRandomSlide(unittest.TestCase):

    def test_slide_shapes(self):
        shape = (20, 100, 100)
        x = np.zeros(shape, dtype='float32')

        # with given output shape
        out_shape = (20, 90, 90)
        trafo_0 = vol.RandomSlide(output_image_size=out_shape[1:])
        x_slided = trafo_0(x)
        self.assertEqual(x_slided.shape, out_shape)

        # with max-misalign
        max_misalign = (10, 10)
        trafo_1 = vol.RandomSlide(max_misalign=max_misalign)
        x_slided = trafo_1(x)
        self.assertEqual(x_slided.shape, out_shape)


def view_random_slide(raw_path, bounding_box, output_shape, raw_key='data', use_misalign=False):
    import h5py
    from volumina_viewer import volumina_n_layer
    with h5py.File(raw_path) as f:
        raw = f[raw_key][bounding_box].astype('float32')

    diff = (
        np.array(raw.shape) - np.array([len(raw), output_shape[0], output_shape[1]]).astype('uint32')
    ) // 2

    if use_misalign:
        max_misalign = tuple(diff)
        trafo = vol.RandomSlide(max_misalign=max_misalign)
    else:
        trafo = vol.RandomSlide(output_shape)
    slided = trafo(raw)

    crop = tuple(slice(diff[i], raw.shape[i] - diff[i]) for i in range(len(diff)))
    cropped = raw[crop]
    assert cropped.shape == slided.shape, "%s, %s" % (str(cropped.shape), str(slided.shape))

    volumina_n_layer([cropped, slided], ['cropped', 'slipped'])


if __name__ == '__main__':
    unittest.main()

    # raw_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/isbi12/train-volume.h5'
    # bb = np.s_[:]
    # shape = (480, 480)
    # view_random_slide(raw_path, bb, shape, use_misalign=False)
