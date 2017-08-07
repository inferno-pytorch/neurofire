import unittest
import neurofire.transforms.segmentation as seg
import numpy as np


class TestSegmentation(unittest.TestCase):

    # generate a random segmentation
    def generate_segmentation(self, generate_2d=True):
        shape = (1, 256, 256) if generate_2d else (1, 8, 256, 256)
        segmentation = np.zeros(shape, dtype='uint32')

        current_label = 1
        if generate_2d:
            for y in range(shape[1]):
                for x in range(shape[2]):
                    segmentation[:, y, x] = current_label
                    if np.random.random() > .8: # change label with 20% probability
                        current_label += 1

        else:
            for z in range(shape[1]):
                for y in range(shape[2]):
                    for x in range(shape[3]):
                        segmentation[:, z, y, x] = current_label
                        if np.random.random() > .8: # change label with 20% probability
                            current_label += 1

        return segmentation


    def generate_toy_data(self):
        segmentation = np.zeros((10, 10), dtype='uint32')
        segmentation[:5, :5] = 1
        segmentation[:5, 5:] = 2
        segmentation[5:, :5] = 3
        segmentation[5:, 5:] = 4

        aff = np.ones((2, 10, 10), dtype='float32')
        aff[0, 5, :] = 0
        aff[1, :, 5] = 0

        # these are the undefined affinities, gonna set them to 0.5 for now
        aff[0, 0, :] = 0.
        aff[1, :, 0] = 0.

        #print(segmentation)
        #print(aff[0])
        #print(aff[1])

        return segmentation, aff


    def affinities_brute_force(self, segmentation, dtype):
        ndim = segmentation.ndim
        affinities = np.zeros((ndim,) + segmentation.shape, dtype=dtype)

        # get the strides (need to divide by bytesize)
        byte_size = np.dtype(dtype).itemsize
        strides = [s // byte_size for s in affinities.strides]

        # iterate over all edges (== 1d affinity coordinates) to get the correct affinities
        for edge in range(affinities.size):

            # translate edge-id to affinity coordinate
            coord = [edge // strides[0]]
            for d in range(1, ndim + 1):
                coord.append((edge % strides[d - 1]) // strides[d])

            # get the corresponding segmentation coordinates
            axis = coord[0]
            coord_u, coord_v = coord[1:], coord[1:]

            # lower the v coordinate if it is valid, otherwise continue
            if(coord_v[axis] > 0):
                coord_v[axis] -= 1
            else:
                continue

            coord = tuple(coord)
            coord_u, coord_v = tuple(coord_u), tuple(coord_v)

            #print(coord_u, coord_v)
            u, v = segmentation[coord_u], segmentation[coord_v]

            # write the correct affinity (0 -> disconnected, 1 -> connected)
            if u == v:
                affinities[coord] = 1
            else:
                affinities[coord] = 0

        return affinities

    def test_brute_force_toy(self):
        segmentation, expected = self.generate_toy_data()
        output = self.affinities_brute_force(segmentation, expected.dtype)
        #print(output[0])
        #print(output[1])
        self.assertEqual(output.shape, expected.shape)
        self.assertTrue((output == expected).all())


    def test_toy(self):
        segmentation, expected = self.generate_toy_data()
        transform = seg.Segmentation2Affinities(dim=2)
        output = transform(segmentation[None, :]).squeeze()
        #print(output[0])
        #print(output[0])
        self.assertEqual(output.shape, expected.shape)
        self.assertTrue((output == expected).all())


    def test_segmentation2affinitiy_random(self):
        # 3D with 3D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 16, 512, 512))
        # Build transform
        transform = seg.Segmentation2Affinities(dim=3)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((16, 512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 3)
        # 3D with 2D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 16, 512, 512))
        # Build transform
        transform = seg.Segmentation2Affinities(dim=2)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((16, 512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 2)
        # 2D with 2D affinities
        wannabe_groundtruth = np.random.uniform(size=(1, 512, 512))
        # Build transform
        transform = seg.Segmentation2Affinities(dim=2)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 2)


    def test_segmentation2affinitiy_2D(self):
        segmentation = self.generate_segmentation()

        # output from the segmentation module
        transform = seg.Segmentation2Affinities(dim=2)
        output = transform(segmentation).squeeze()

        # brute force loop
        output_expected = self.affinities_brute_force(segmentation.squeeze(), output.dtype)

        self.assertEqual(output.shape, output_expected.shape)
        self.assertTrue((output == output_expected).all())


    def test_segmentation2affinitiy_3D(self):
        segmentation = self.generate_segmentation(False)

        # output from the segmentation module
        transform = seg.Segmentation2Affinities(dim=3)
        output = transform(segmentation).squeeze()

        # brute force loop
        output_expected = self.affinities_brute_force(segmentation.squeeze(), output.dtype)

        self.assertEqual(output.shape, output_expected.shape)
        self.assertTrue((output == output_expected).all())


    def test_cc_2d(self):
        x = np.array(
            [[1, 1, 0, 1, 1],
             [1, 1, 0, 0, 1],
             [1, 0, 1, 0, 0],
             [1, 0, 1, 1, 0],
             [0, 1, 1, 1, 1]],
            dtype='uint32'
        )
        for label_segmentation in (False, True):
            cc = seg.ConnectedComponents2D(label_segmentation=label_segmentation)
            y = cc(x)
            self.assertEqual(y.shape, x.shape)
            uniques = np.unique(y).tolist()
            self.assertEqual(uniques, [0, 1, 2, 3])


    def test_cc_3d(self):
        x = np.array(
            [[[1, 1, 0],
              [1, 0, 0],
              [0, 0, 1]],
             [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 0]],
             [[1, 0, 1],
              [0, 0, 0],
              [0, 0, 0]]],
            dtype='uint32'
        )
        for label_segmentation in (False, True):
            cc = seg.ConnectedComponents3D(label_segmentation=label_segmentation)
            y = cc(x)
            self.assertEqual(y.shape, x.shape)
            uniques = np.unique(y).tolist()
            self.assertEqual(uniques, [0, 1, 2, 3, 4])


    def test_cc_label_segmentation(self):
        x = np.array(
            [[1, 2, 1],
             [1, 1, 1],
             [1, 2, 1]],
            dtype='uint32'
        )
        cc = seg.ConnectedComponents2D(label_segmentation=True)
        y = cc(x)
        self.assertEqual(y.shape, x.shape)
        uniques = np.unique(y).tolist()
        self.assertEqual(uniques, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
