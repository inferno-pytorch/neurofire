import vigra
from volumina_viewer import volumina_n_layer


def view_example(path, obj_id):
    mask = vigra.readHDF5(path, 'fmobject_%s/object_mask' % obj_id)
    seg = vigra.readHDF5(path, 'fmobject_%s/seg' % obj_id)
    gt = vigra.readHDF5(path, 'fmobject_%s/gt' % obj_id)
    edges = vigra.readHDF5(path, 'fmobject_%s/edge_mask' % obj_id)
    target = vigra.readHDF5(path, 'fmobject_%s/target' % obj_id)

    volumina_n_layer([seg, gt, mask, edges, target],
                     ['seg', 'gt', 'mask', 'edges', 'target'])


if __name__ == '__main__':
    path = './artificial_fm_targets_sampleA.h5'
    view_example(path, "1_12")
