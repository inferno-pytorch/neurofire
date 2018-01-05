import unittest
import numpy as np
import torch


class TestSegmentation(unittest.TestCase):

    # generate a random segmentation
    @staticmethod
    def generate_segmentation(shape):
        segmentation = np.zeros(shape, dtype='uint32')
        label = 0
        for index in range(segmentation.size):
            coord = np.unravel_index(index, shape)
            segmentation[coord] = label
            if np.random.random() > .9:  # change label with 10% probability
                label += 1
        return segmentation

    # generate a toy segmentation
    @staticmethod
    def generate_toy_data():
        seg = np.zeros((10, 10), dtype='uint32')
        seg[:5, :5] = 1
        seg[:5, 5:] = 2
        seg[5:, :5] = 3
        seg[5:, 5:] = 4

        aff = np.ones((2, 10, 10), dtype='float32')
        aff[0, 5, :] = 0
        aff[1, :, 5] = 0

        # these are the undefined affinities, gonna set them to 0.5 for now
        aff[0, 0, :] = 0.
        aff[1, :, 0] = 0.
        return seg, aff

    @staticmethod
    def affinities_brute_force(segmentation, offsets):
        shape = segmentation.shape
        affinities = np.zeros((len(offsets),) + shape, dtype='float32')

        # iterate over all edges (== 1d affinity coordinates) to get the correct affinities
        for edge in range(affinities.size):
            aff_coord = np.unravel_index(edge, affinities.shape)
            offset = offsets[aff_coord[0]]
            coord_u = aff_coord[1:]
            coord_v = tuple(cu + off for cu, off in zip(coord_u, offset))

            # check if coord_v is valid
            if any(cv < 0 or cv >= sha for cv, sha in zip(coord_v, shape)):
                continue

            # write the correct affinity (0 -> disconnected, 1 -> connected)
            u, v = segmentation[coord_u], segmentation[coord_v]
            affinities[aff_coord] = 1. if u == v else 0.

        return affinities

    def test_seg2mem(self):
        from neurofire.transform.segmentation import Segmentation2Membranes
        trafo = Segmentation2Membranes()
        for dim in (2, 3):
            shape = dim * (128,)
            seg = self.generate_segmentation(shape)
            membranes = trafo(seg)
            self.assertEqual(membranes.shape, shape)
            # make for torch tensor, check that agree
            # membranes_torch =
            # self.assertEqual(membranes_torch.shape, shape)

    def test_brute_force_affs_toy(self):
        offsets = [(-1, 0), (0, -1)]
        segmentation, expected = self.generate_toy_data()
        output = self.affinities_brute_force(segmentation, offsets)
        self.assertEqual(output.shape, expected.shape)
        self.assertTrue(np.allclose(output, expected))

    def test_affs_toy(self):
        from neurofire.transform.segmentation import Segmentation2Affinities
        segmentation, expected = self.generate_toy_data()
        transform = Segmentation2Affinities(dim=2)
        output = transform(segmentation[None, :]).squeeze()
        self.assertEqual(output.shape, expected.shape)
        self.assertTrue(np.allclose(output, expected))

    def test_affs_random(self):
        from neurofire.transform.segmentation import Segmentation2Affinities

        # 3D with 3D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 16, 512, 512))
        # Build transform
        transform = Segmentation2Affinities(dim=3)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((16, 512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 3)

        # 3D with 2D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 16, 512, 512))
        # Build transform
        transform = Segmentation2Affinities(dim=2)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((16, 512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 2)

        # 2D with 2D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 512, 512))
        # Build transform
        transform = Segmentation2Affinities(dim=2)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 2)

    # FIXME outputs do not agree 100 %
    def test_affs_2d_orders(self):
        from neurofire.transform.segmentation import Segmentation2Affinities
        shape = (1, 64, 64)
        segmentation = self.generate_segmentation(shape)

        def order_to_offsets(order):
            return [(-order, 0), (0, -order)]

        for order in (1, 2, 3, 5, 7, 20):
            print("2D Test: Checking order", order)
            # output from the segmentation module
            transform = Segmentation2Affinities(dim=2, order=order)
            output = transform(segmentation).squeeze()

            output_expected = self.affinities_brute_force(segmentation.squeeze(),
                                                          order_to_offsets(order))

            self.assertEqual(output.shape, output_expected.shape)
            print(output.shape)
            print(np.sum(np.isclose(output, output_expected)), '/', output.size)
            where = np.where(np.logical_not(np.isclose(output, output_expected)))
            print(where)
            print(output[where], output_expected[where])
            self.assertTrue(np.allclose(output, output_expected))

    # FIXME outputs do not agree 100 %
    def test_affs_3d_orders(self):
        from neurofire.transform.segmentation import Segmentation2Affinities
        shape = (1, 64, 64, 64)
        segmentation = self.generate_segmentation(shape)

        def order_to_offsets(order):
            return [(-order, 0, 0), (0, -order, 0), (0, 0, -order)]

        for order in (1, 2, 3, 5, 7, 20):
            print("3D Test: Checking order", order)
            # output from the segmentation module
            transform = Segmentation2Affinities(dim=3, order=order)
            output = transform(segmentation).squeeze()

            output_expected = self.affinities_brute_force(segmentation.squeeze(),
                                                          order_to_offsets(order))

            self.assertEqual(output.shape, output_expected.shape)
            print(np.sum(np.isclose(output, output_expected)), '/', output.size)
            self.assertTrue(np.allclose(output, output_expected))

    # FIXME outputs do not agree 100 %
    # TODO test for 2D
    def test_affs_from_offsets_3D(self):
        from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets
        shape = (1, 64, 64, 64)
        segmentation = self.generate_segmentation(shape).astype('int32')
        seg_torch = torch.from_numpy(segmentation)

        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (3, 0, 0), (0, 3, 0), (0, 0, 3),
                   (1, 1, 0), (-1, 1, 0)]

        trafo = Segmentation2AffinitiesFromOffsets(3, offsets,
                                                   add_singleton_channel_dimension=False)
        output = trafo(seg_torch).numpy()
        output_expected = self.affinities_brute_force(segmentation.squeeze(), offsets)
        self.assertEqual(output.shape, output_expected.shape)
        print(np.sum(np.isclose(output, output_expected)), '/', output.size)
        self.assertTrue(np.allclose(output, output_expected))

    def test_cc_2d(self):
        from neurofire.transform.segmentation import ConnectedComponents2D
        x = np.array([[1, 1, 0, 1, 1],
                      [1, 1, 0, 0, 1],
                      [1, 0, 1, 0, 0],
                      [1, 0, 1, 1, 0],
                      [0, 1, 1, 1, 1]],
                     dtype='uint32')
        for label_segmentation in (False, True):
            cc = ConnectedComponents2D(label_segmentation=label_segmentation)
            y = cc(x)
            self.assertEqual(y.shape, x.shape)
            uniques = np.unique(y).tolist()
            self.assertEqual(uniques, [0, 1, 2, 3])

    def test_cc_3d(self):
        from neurofire.transform.segmentation import ConnectedComponents3D
        x = np.array([[[1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1]],
                      [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]],
                      [[1, 0, 1],
                       [0, 0, 0],
                       [0, 0, 0]]],
                     dtype='uint32')
        for label_segmentation in (False, True):
            cc = ConnectedComponents3D(label_segmentation=label_segmentation)
            y = cc(x)
            self.assertEqual(y.shape, x.shape)
            uniques = np.unique(y).tolist()
            self.assertEqual(uniques, [0, 1, 2, 3, 4])

    def test_cc_label_segmentation(self):
        from neurofire.transform.segmentation import ConnectedComponents2D
        x = np.array([[1, 2, 1],
                      [1, 1, 1],
                      [1, 2, 1]],
                     dtype='uint32')
        cc = ConnectedComponents2D(label_segmentation=True)
        y = cc(x)
        self.assertEqual(y.shape, x.shape)
        uniques = np.unique(y).tolist()
        self.assertEqual(uniques, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
