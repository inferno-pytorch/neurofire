import numpy as np
import vigra
import h5py
import nifty
from neurofire.transform.false_merge_gt import ArtificialFalseMerges


# TODO implement actual unittest
def test_false_merge_gt():
    from cremi_tools.viewer.volumina import view
    path = '/home/papec/Work/neurodata_hdd/cremi/sample_A_20160501.hdf'
    with h5py.File(path) as f:
        labels = f['volumes/labels/neuron_ids'][:]
        vigra.analysis.relabelConsecutive(labels, out=labels)
        raw = f['volumes/raw'][:]

    trafo = ArtificialFalseMerges(target_distances=(5., 15., 25.))
    shape = raw.shape
    block_shape = [50, 512, 512]
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_samples = 1
    blocks = np.random.choice(np.arange(blocking.numberOfBlocks), n_samples)
    for block_id in blocks:
        block = blocking.getBlock(block_id)
        bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
        raw_sub, labels_sub = raw[bb], labels[bb]
        inputs = [raw_sub, labels_sub]
        inputs, labels_merged = trafo(inputs)
        raw_new = inputs[0]
        mask = inputs[1].astype('uint32')
        view([raw_new, mask, labels_sub,
              labels_merged.transpose((1, 2, 3, 0))])


def false_merge_gt_stresstest():
    path = '/home/papec/Work/neurodata_hdd/cremi/sampleA/sampleA_neurongt_none.h5'
    with h5py.File(path) as f:
        labels = f['data'][:].astype('uint64')
        vigra.analysis.relabelConsecutive(labels, out=labels)
    raw_path = '/home/papec/Work/neurodata_hdd/cremi/sampleA/sampleA_raw_none.h5'
    with h5py.File(raw_path) as f:
        raw = f['data'][:]
    assert raw.shape == labels.shape

    # path = '/home/papec/Work/neurodata_hdd/cremi/sample_A_20160501.hdf'
    # with h5py.File(path) as f:
    #     labels = f['volumes/labels/neuron_ids'][:]
    #     vigra.analysis.relabelConsecutive(labels, out=labels)
    #     raw = f['volumes/raw'][:]

    print("Dtype:")
    print(labels.dtype)
    print("Dtype:")

    trafo = ArtificialFalseMerges(target_distances=(5., 15., 25.))
    shape = labels.shape
    block_shape = [50, 512, 512]
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_samples = 250
    blocks = np.random.choice(np.arange(blocking.numberOfBlocks), n_samples)
    for ii, block_id in enumerate(blocks):
        print(ii, '/', n_samples)
        block = blocking.getBlock(block_id)
        bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
        raw_sub, labels_sub = raw[bb], labels[bb]
        inputs, labels_merged = trafo(raw_sub, labels_sub)
        print(inputs.shape)
        print(labels_merged.shape)
        # raw_new = inputs[0]
        # mask = inputs[1].astype('uint32')
        # view([raw_new, mask, labels_sub,
        #       labels_merged.transpose((1, 2, 3, 0))])


if __name__ == '__main__':
    false_merge_gt_stresstest()
