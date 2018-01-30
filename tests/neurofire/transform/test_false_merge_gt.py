import numpy as np
import vigra
import h5py
import nifty
from neurofire.transform.false_merge_gt import ArtificialFalseMerges


# TODO implement actual unittest
def test_false_merge_gt():
    from cremi_tools.viewer.volumina import view
    path = '/home/papec/Work/neurodata_hdd/cremi/sample_B_20160501.hdf'
    with h5py.File(path) as f:
        labels = f['volumes/labels/neuron_ids'][:]
        vigra.analysis.relabelConsecutive(labels, out=labels)
        raw = f['volumes/raw'][:]

    trafo = ArtificialFalseMerges(target_distances=(1., 5., 15., 25.),
                                  crop_to_object=(1, 27, 27))

    shape = raw.shape
    block_shape = [75, 756, 756]
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_samples = 1
    blocks = np.random.choice(np.arange(blocking.numberOfBlocks), n_samples)
    for block_id in blocks:
        block = blocking.getBlock(block_id)
        bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
        raw_sub, labels_sub = raw[bb], labels[bb]
        inputs, labels_merged, bb = trafo(raw_sub, labels_sub, return_bounding_box=True)
        labels_sub = labels_sub[bb]
        raw_new = inputs[0]
        mask = inputs[1].astype('uint32')
        print(inputs.shape, labels_merged.shape)
        view([raw_new, mask, labels_sub,
              labels_merged.transpose((1, 2, 3, 0))])


def false_merge_gt_stresstest():
    # path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi/sampleA/gt/sampleA_neurongt_none.h5'
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sampleA/gt/sampleA_neurongt_none.h5'
    with h5py.File(path) as f:
        labels = f['data'][:]
    raw_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sampleA/raw/sampleA_raw_none.h5'
    with h5py.File(raw_path) as f:
        raw = f['data'][:]
    assert raw.shape == labels.shape

    trafo = ArtificialFalseMerges(target_distances=(5., 15., 25.))
    shape = labels.shape
    block_shape = [50, 512, 512]
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_samples = 25
    blocks = np.random.choice(np.arange(blocking.numberOfBlocks), n_samples)
    for ii, block_id in enumerate(blocks):
        print(ii, '/', n_samples)
        block = blocking.getBlock(block_id)
        bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
        raw_sub, labels_sub = raw[bb], labels[bb]
        inputs, labels_merged = trafo(raw_sub, labels_sub)
        assert inputs.shape[1:] == labels_merged.shape[1:], "%s, %s" % (str(inputs.shape), str(labels_merged.shape))
        # raw_new = inputs[0]
        # mask = inputs[1].astype('uint32')
        # view([raw_new, mask, labels_sub,
        #       labels_merged.transpose((1, 2, 3, 0))])


if __name__ == '__main__':
    test_false_merge_gt()
    # false_merge_gt_stresstest()
