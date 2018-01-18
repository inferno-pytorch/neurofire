import numpy as np
import vigra
import nifty.graph.rag as nrag
import h5py


def obj_mask_and_bb(obj_id, seg):
    obj_coords = np.where(seg == obj_id)
    bb = tuple(slice(coord.min(), coord.max() + 1) for coord in obj_coords)
    mask = seg[bb]
    mask = mask == obj_id
    return mask, bb


def make_merge(obj_a, obj_b, gt, out_file, raw):
    gt_merged = gt.copy()
    gt_merged[gt_merged == obj_b] = obj_a
    mask, bb = obj_mask_and_bb(obj_a, gt_merged)

    obj_id = "_%i_%i" % (min(obj_a, obj_b), max(obj_a, obj_b))
    # we overcount so must avoid duplicates
    group = 'fmobject%s' % obj_id
    if group in out_file:
        return

    gt_bb = gt[bb]
    gt_bb, _, mapping = vigra.analysis.relabelConsecutive(gt_bb)
    obj_a, obj_b = mapping[obj_a], mapping[obj_b]

    # build rag and edge builder
    rag = nrag.gridRag(gt_bb, numberOfThreads=8)
    edge_builder = nrag.ragCoordinates(rag, numberOfThreads=8)
    fm_edge = rag.findEdge(obj_a, obj_b)

    edges = np.zeros(rag.numberOfEdges, dtype='uint32')
    edges[fm_edge] = 1
    edge_vol = edge_builder.edgesToVolume(edges, numberOfThreads=8)
    # apply distance transform to get the fm indicators
    dist_from_fm = vigra.filters.distanceTransform(edge_vol, pixel_pitch=[10., 1., 1.])
    dist_thresh = 25
    fm_target = (dist_from_fm < dist_thresh).astype('float32')
    # bound to object mask
    fm_target[np.logical_not(mask)] = 0

    out_file.create_dataset('fmobject%s/target' % obj_id, data=fm_target, compression='gzip')
    out_file.create_dataset('fmobject%s/object_mask' % obj_id, data=mask.astype('float32'), compression='gzip')
    out_file.create_dataset('fmobject%s/gt' % obj_id, data=gt_bb.astype('uint32'), compression='gzip')
    out_file.create_dataset('fmobject%s/seg' % obj_id, data=gt_merged[bb].astype('uint32'), compression='gzip')
    out_file.create_dataset('fmobject%s/edge_mask' % obj_id, data=edge_vol.astype('uint32'), compression='gzip')
    out_file.create_dataset('fmobject%s/raw' % obj_id, data=raw[bb].astype('uint8'), compression='gzip')


def make_artificial_merges(gt, raw, out_path, percentile=90):
    # prepare data and rag
    ignore_mask = gt == 0
    vigra.analysis.relabelConsecutive(gt, start_label=1)
    gt[ignore_mask] = 0
    gt_rag = nrag.gridRag(gt, numberOfThreads=8)
    # find candidate objects for merges
    gt_objs, counts = np.unique(gt, return_counts=True)
    print("Number of objects", len(gt_objs))
    size_threshold = np.percentile(counts, percentile)
    print("Min size", size_threshold)
    assert size_threshold > 1000
    candidates = gt_objs[counts > size_threshold]
    # get rid of ignore label
    candidates = candidates[candidates != 0]
    # choose from the candidates
    example_count = 0
    out_file = h5py.File(out_path, 'w')
    for candidate_obj in candidates:
        adjacent_objs = np.array([adj[0]
                                  for adj in gt_rag.nodeAdjacency(candidate_obj)], dtype='uint32')
        candidate_merges = adjacent_objs[np.in1d(adjacent_objs, candidates)]
        if candidate_merges.size == 0:
            continue
        for merge_obj in candidate_merges:
            print("Making false merge for objects", candidate_obj, merge_obj)
            merge_obj = np.random.choice(candidate_merges, size=1)[0]
            make_merge(candidate_obj, merge_obj, gt, out_file, raw)
    out_file.close()



if __name__ == '__main__':
    for sample in ('A', 'B', 'C'):
        gt = vigra.readHDF5('/media/papec/data/papec/cremi_gt/sample%s_neurongt.h5'
                             % sample, 'data')
        raw = vigra.readHDF5('/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sample%s/raw/sample%s_raw_automatically_realigned.h5'
                             % (sample, sample), 'data')
        out_file = './artificial_fm_targets_sample%s.h5' % sample
        make_artificial_merges(gt, raw, out_file)
