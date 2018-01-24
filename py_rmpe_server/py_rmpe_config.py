#!/usr/bin/env python

import numpy as np

def ltr_parts(parts_dict):
    # when we flip image left parts became right parts and vice versa. This is the list of parts to exchange each other.
    leftParts  = [ parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"] ]
    rightParts = [ parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"] ]
    return leftParts,rightParts


class RmpeGlobalConfig:

    width = 368
    height = 368

    stride = 8

    parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]
    num_parts = len(parts)
    parts_dict = dict(zip(parts, range(num_parts)))
    parts += ["background"]
    num_parts_with_background = len(parts)

    leftParts, rightParts = ltr_parts(parts_dict)

    # this numbers probably copied from matlab they are 1.. based not 0.. based
    limb_from = [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]
    limb_to = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
    limbs_conn = zip(limb_from, limb_to)
    limbs_conn = [(fr - 1, to - 1) for (fr, to) in limbs_conn]

    paf_layers = 2*len(limbs_conn)
    heat_layers = num_parts
    num_layers = paf_layers + heat_layers + 1

    paf_start = 0
    heat_start = paf_layers
    bkg_start = paf_layers + heat_layers

    data_shape = (3, height, width)     # 3, 368, 368
    mask_shape = (height//stride, width//stride)  # 46, 46
    parts_shape = (num_layers, height//stride, width//stride)  # 57, 46, 46

class TransformationParams:

    target_dist = 0.6;
    scale_prob = 1;   # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
    scale_min = 0.5;
    scale_max = 1.1;
    max_rotate_degree = 40.
    center_perterb_max = 40.
    flip_prob = 0.5
    sigma = 7.
    paf_thre = 8.  # it is original 1.0 * stride in this program


class RmpeCocoConfig:


    parts = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
     'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank',
     'Rank']

    num_parts = len(parts)

    # for COCO neck is calculated like mean of 2 shoulders.
    parts_dict = dict(zip(parts, range(num_parts)))

    @staticmethod
    def convert(joints):

        result = np.zeros((joints.shape[0], RmpeGlobalConfig.num_parts, 3), dtype=np.float)
        result[:,:,2]=2.  # 2 - abstent, 1 visible, 0 - invisible

        for p in RmpeCocoConfig.parts:
            coco_id = RmpeCocoConfig.parts_dict[p]
            global_id = RmpeGlobalConfig.parts_dict[p]
            assert global_id!=1, "neck shouldn't be known yet"
            result[:,global_id,:]=joints[:,coco_id,:]

        neckG = RmpeGlobalConfig.parts_dict['neck']
        RshoC = RmpeCocoConfig.parts_dict['Rsho']
        LshoC = RmpeCocoConfig.parts_dict['Lsho']


        # no neck in coco database, we calculate it as averahe of shoulders
        # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        both_shoulders_known = (joints[:, LshoC, 2]<2)  &  (joints[:, RshoC, 2]<2)
        result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
                                                    joints[both_shoulders_known, LshoC, 0:2]) / 2
        result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
                                                                 joints[both_shoulders_known, LshoC, 2])

        return result

class RpmeMPIIConfig:

    parts = ["HeadTop", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
             "RAnkle", "LHip", "LKnee", "LAnkle"]

    numparts = len(parts)

    #14 - Chest is calculated like "human center location provided by the annotated data"


    @staticmethod
    def convert(joints):
        raise "Not implemented"



# more information on keypoints mapping is here
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/7


def check_layer_dictionary():

    dct = RmpeGlobalConfig.parts[:]
    dct = [None]*(RmpeGlobalConfig.num_layers-len(dct)) + dct

    for (i,(fr,to)) in enumerate(RmpeGlobalConfig.limbs_conn):
        name = "%s->%s" % (RmpeGlobalConfig.parts[fr], RmpeGlobalConfig.parts[to])
        print(i, name)
        x = i*2
        y = i*2+1

        assert dct[x] is None
        dct[x] = name + ":x"
        assert dct[y] is None
        dct[y] = name + ":y"

    print(dct)


if __name__ == "__main__":
    check_layer_dictionary()

