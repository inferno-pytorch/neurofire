import unittest
import neurofire.transforms.volume as vol
import numpy as np
import h5py
# import torch


# TODO proper tests on random data
class TestRandomSlide(unittest.TestCase):

    def test_slide_shapes(self):
        shape = (20, 100, 100)
        x = np.zeros(shape, dtype='float32')
        out_shape = (20, 90, 90)
        trafo = vol.RandomSlide(out_shape[1:])
        x_slided = trafo(x)
        self.assertEqual(x_slided.shape, out_shape)


def view_random_slide(raw_path, bounding_box, output_shape, raw_key='data'):
    from volumina_viewer import volumina_n_layer
    with h5py.File(raw_path) as f:
        raw = f[raw_key][bounding_box].astype('float32')

    trafo = vol.RandomSlide(output_shape)
    slided = trafo(raw)

    diff = (
        np.array(raw.shape) - np.array([len(raw), output_shape[0], output_shape[1]]).astype('uint32')
    ) // 2
    crop = tuple(slice(diff[i], raw.shape[i] - diff[i]) for i in range(len(diff)))
    cropped = raw[crop]
    assert cropped.shape == slided.shape, "%s, %s" % (str(cropped.shape), str(slided.shape))

    volumina_n_layer([cropped, slided], ['cropped', 'slipped'])


if __name__ == '__main__':
    unittest.main()
    #raw_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/isbi12/train-volume.h5'
    #bb = np.s_[:]
    #shape = (480, 480)
    #view_random_slide(raw_path, bb, shape)
