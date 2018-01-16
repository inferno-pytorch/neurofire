import json
import numpy as np
import h5py
import vigra
# import nifty.graph.rag as nrag
import nifty.groun_dtruth as ngt


def get_false_merge_objects(gt, seg,
                            size_threshold,
                            overlap_threshold):
    # rag = nrag.gridRag(seg, numberOfThreads=8)
    overlaps = ngt.overlaps(seg, gt)

    objects, obj_counts = np.unique(seg, return_counts=True)
    objects = objects[objects > size_threshold]

    # TODO TP
    fm_objects = {}
    for obj_id in objects:
        fm_gt_ids = check_for_fm(obj_id, overlaps, overlap_threshold)
        if fm_gt_ids:
            fm_objects[obj_id] = fm_gt_ids
    return fm_objects


def check_for_fm(obj_id, overlaps, overlap_threshold):
    ovlp_gt_ids, ovlp_percentage = overlaps.overlapArrayNormalized(obj_id, sorted=True)
    # TODO find proper criterion to determine false merge gt-ids
    if ovlp_percentage.max() < overlap_threshold:
        return
    else:
        return []


def obj_mask_and_bb(obj_id, seg):
    obj_coords = np.where(seg == obj_id)
    bb = tuple(slice(coord.min(), coord.max() + 1) for coord in obj_coords)
    mask = seg[bb]
    mask = mask == obj_id
    return mask, bb


def make_fm_groundtruth(fm_objects, gt, seg, hmap, out_path):
    out_file = h5py.File(out_path, 'w')
    for obj_id, fm_gt_ids in fm_objects.items():
        mask, bb = obj_mask_and_bb(obj_id, seg)
        hmap_bb, gt_bb = hmap[bb], gt[bb]
        gt_mask = np.ma.masked_array(gt_bb, np.in1d(fm_gt_ids, gt_bb)).mask
        gt_bb[np.logical_not(gt_mask)] = 0
        gt_bb, _ = vigra.analysis.watershedsNew(hmap_bb, seeds=gt_bb)
        gt_bb[np.logial_not(mask)] = 0
        out_file.create_dataset('fmobject%i/')

    out_file.close()


if __name__ == '__main__':
    sample = 'A'
    seg = vigra.readHDF5('%s' % sample, 'data')
    gt = vigra.readHDF5('%s' % sample, 'data')
    fm_objects = get_false_merge_objects(gt, seg)
    with open('./fm_ids.json', 'w') as f:
        json.dump(fm_objects, f)
