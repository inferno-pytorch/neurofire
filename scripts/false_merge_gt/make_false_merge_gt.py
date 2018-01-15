import numpy as np
import vigra
import nifty.graph.rag as nrag
import nifty.groundtruth as ngt


def make_false_merge_groundtruth(sample, size_threshold):
    seg = vigra.readHDF5('%s' % sample, 'data')
    gt = vigra.readHDF5('%s' % sample, 'data')

    rag = nrag.gridRag(seg, numberOfThreads=8)
    objects, obj_counts = np.unique(seg, return_counts=True)


def check_for_fm():
    pass


if __name__ == '__main__':
    sample = 'A'
    make_false_merge_groundtruth(sample)
