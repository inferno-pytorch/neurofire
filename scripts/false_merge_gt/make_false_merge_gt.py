import json
from itertools import combinations
import numpy as np
import h5py
import vigra
import nifty.graph.rag as nrag
import nifty.ground_truth as ngt


def get_false_merge_objects(gt, seg,
                            size_percentile,
                            overlap_threshold):
    # rag = nrag.gridRag(seg, numberOfThreads=8)
    overlaps = ngt.overlap(seg, gt)

    objects, obj_counts = np.unique(seg, return_counts=True)
    size_threshold = np.percentile(obj_counts, size_percentile)
    print("Min-size:", size_threshold)
    objects = objects[obj_counts > size_threshold]

    # TODO TP
    fm_objects = {}
    for ii, obj_id in enumerate(objects):
        # print("Checking object", ii, "/", len(objects))
        fm_gt_ids = check_for_fm(obj_id, overlaps, overlap_threshold)
        if fm_gt_ids:
            fm_objects[int(obj_id)] = [int(gt_id) for gt_id in fm_gt_ids]
    return fm_objects


def check_for_fm(obj_id, overlaps, overlap_threshold):
    ovlp_gt_ids, ovlp_percentage = overlaps.overlapArraysNormalized(obj_id)
    # remove ignore label if present and
    # renormalize percentages
    if 0 in ovlp_gt_ids:
        mask = ovlp_gt_ids != 0
        ovlp_gt_ids = ovlp_gt_ids[mask]
        ovlp_percentage[mask]
        normalisation = np.sum(ovlp_percentage)
        ovlp_percentage /= normalisation
        assert np.isclose(np.sum(ovlp_percentage), 1.)

    # we consider an object to contain a false merge,
    # if more than one of the enclosed gt-objects are above the
    # ovlp threshold
    above_thresh = ovlp_percentage > overlap_threshold
    if np.sum(above_thresh) > 1:
        return ovlp_gt_ids[above_thresh].tolist()
    else:
        return []


def obj_mask_and_bb(obj_id, seg):
    obj_coords = np.where(seg == obj_id)
    bb = tuple(slice(coord.min(), coord.max() + 1) for coord in obj_coords)
    mask = seg[bb]
    mask = mask == obj_id
    print(mask.shape)
    return mask, bb


def make_fm_groundtruth(fm_objects, gt, seg, out_path, dist_thresh=25):
    out_file = h5py.File(out_path, 'w')

    # iterate over gt false merges
    for obj_id, fm_gt_ids in fm_objects.items():
        obj_id = int(obj_id)
        print("extracting target for", obj_id)
        # mask and bounding box of current object
        mask, bb = obj_mask_and_bb(obj_id, seg)
        if mask.shape[0] == 1:
            continue
        # map gt to the bounding box and relabel
        gt_bb = gt[bb]
        gt_bb, _, mapping = vigra.analysis.relabelConsecutive(gt_bb)
        fm_ids = [mapping[idd] for idd in fm_gt_ids]
        # build rag and edge builder
        rag = nrag.gridRag(gt_bb, numberOfThreads=8)
        edge_builder = nrag.ragCoordinates(rag, numberOfThreads=8)
        # find the coordinates of the fm-edges
        node_pairs = combinations(fm_ids, 2)
        fm_edges = [rag.findEdge(np[0], np[1]) for np in node_pairs]
        fm_edges = tuple(fme for fme in fm_edges if fme != -1)
        if not fm_edges:
            print("Skipping object with edges that are not touching:", obj_id)
            continue
        edges = np.zeros(rag.numberOfEdges, dtype='uint32')
        edges[fm_edges] = 1
        edge_vol = edge_builder.edgesToVolume(edges, numberOfThreads=8)
        # apply distance transform to get the fm indicators
        dist_from_fm = vigra.filters.distanceTransform(edge_vol, pixel_pitch=[10., 1., 1.])
        fm_target = (dist_from_fm < dist_thresh).astype('float32')
        # bound to object mask
        fm_target[np.logical_not(mask)] = 0

        out_file.create_dataset('fmobject%i/target' % obj_id, data=fm_target, compression='gzip')
        out_file.create_dataset('fmobject%i/object_mask' % obj_id, data=mask.astype('float32'), compression='gzip')
        out_file.create_dataset('fmobject%i/gt' % obj_id, data=gt_bb.astype('uint32'), compression='gzip')
        out_file.create_dataset('fmobject%i/seg' % obj_id, data=seg[bb].astype('uint32'), compression='gzip')
        out_file.create_dataset('fmobject%i/edge_mask' % obj_id, data=edge_vol.astype('uint32'), compression='gzip')
    out_file.close()


if __name__ == '__main__':
    sample = 'A'
    gt = vigra.readHDF5('/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sample%s/gt/sample%s_neurongt_automatically_realignedV2.h5'
                         % (sample, sample),
                         'data')
    # TODO dunno if this is the proper segmentation that should be used, but doesn't matter for testing
    seg = vigra.readHDF5('/media/papec/data/papec/multicuts/mc_seg_sample%s.h5'
                         % sample, 'data')
    assert seg.shape == gt.shape, "%s, %s" % (str(seg.shape), str(gt.shape))

    have_objs = False
    if have_objs:
        with open('./fm_ids_sample%s.json' % sample, 'r') as f:
            fm_objects = json.load(f)

    else:
        fm_objects = get_false_merge_objects(gt, seg,
                                             size_percentile=80,
                                             overlap_threshold=0.3)

        print("Found", len(fm_objects), "objects with false merges")
        with open('./fm_ids_sample%s.json' % sample, 'w') as f:
            json.dump(fm_objects, f)

    make_fm_groundtruth(fm_objects, gt, seg, out_path='./fm_targets_sample%s.h5' % sample)
