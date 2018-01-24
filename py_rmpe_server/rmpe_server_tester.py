#!/usr/bin/env python

import sys
import os
sys.path.append("..")

from time import time
from training.ds_generators import DataGeneratorClient

import cv2
import numpy as np

from py_rmpe_config import RmpeGlobalConfig

mask_pattern = np.zeros((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 3), dtype=np.uint8)
heat_pattern = np.zeros((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 3), dtype=np.uint8)
mask_y, mask_x = np.mgrid[0:RmpeGlobalConfig.height, 0:RmpeGlobalConfig.width]
grid = (mask_x//8 % 2) + (mask_y//8 % 2)

mask_pattern[grid==1]=(255,255,255)
mask_pattern[grid!=1]=(128,128,128)

heat_pattern[...] = (0,0,255)

save_to = 'augmented'  # save new server to output, c++ server to original and compare images


def save_images(num, image, mask, paf):

    image = image.transpose([1,2,0])

    mask_img = mask.reshape((mask.shape[0], mask.shape[1], 1))
    mask_img = (mask_img*255).astype(np.uint8)
    mask_img = cv2.resize(mask_img, (RmpeGlobalConfig.height, RmpeGlobalConfig.width), interpolation=cv2.INTER_NEAREST)
    mask_img = mask_img.reshape((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 1))

    masked_img = image.copy()
    masked_img = masked_img*(mask_img/255.0) + mask_pattern*(1.-mask_img/255.0)

    os.makedirs(save_to, exist_ok=True)

    #cv2.imwrite(save_to+"/%07d%s.png" % (num, ""), image)
    #cv2.imwrite(save_to+"/%07d%s.png" % (num, "mask"), mask_img)
    cv2.imwrite(save_to + "/%07d%s.png" % (num, "masked"), masked_img)

    parts = []

    for i in range(RmpeGlobalConfig.num_parts_with_background):
        heated_image = image.copy()

        heat_img = paf[RmpeGlobalConfig.heat_start+i]

        heat_img = cv2.resize(heat_img, (RmpeGlobalConfig.height, RmpeGlobalConfig.width), interpolation=cv2.INTER_NEAREST)
        heat_img = heat_img.reshape((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 1))

        heated_image = heated_image*(1-heat_img) + heat_pattern*heat_img

        parts += [heated_image]

    parts = np.vstack(parts)
    cv2.imwrite(save_to+"/%07d%s.png" % (num, "heat"), parts)


    pafs = []
    stride = RmpeGlobalConfig.stride

    for i,(fr,to) in enumerate(RmpeGlobalConfig.limbs_conn):
        paffed_image = image.copy()

        pafX = paf[RmpeGlobalConfig.paf_start + i * 2]
        pafY = paf[RmpeGlobalConfig.paf_start + i * 2 + 1]

        for x in range(RmpeGlobalConfig.width//stride):
            for y in range(RmpeGlobalConfig.height//stride):
                X = pafX[y, x]
                Y = pafY[y, x]

                if X!=0 or Y!=0:
                    cv2.arrowedLine(paffed_image, (x*stride,y*stride), (int(x*stride+X*stride),int(y*stride+Y*stride)), color=(0,0,255), thickness=1, tipLength=0.5)

        pafs += [paffed_image]


    pafs = np.vstack(pafs)
    cv2.imwrite(save_to+"/%07d%s.png" % (num, "paf"), pafs)



def time_processed(client, batch_size):

    num = 0
    start = time()

    for x,y in client.gen():
        num += 1
        elapsed = time() - start
        print(num*batch_size, num*batch_size/elapsed, [ i.shape for i in x ], [i.shape for i in y] )

def time_raw(client, save):

    num = 0
    start = time()

    for foo in client.gen_raw():

        if len(foo) == 3:
            x, y, z = foo
        elif len(foo) == 4:
            x, y, z, k = foo
        else:
            raise NotImplementedError("Unknown number of tensors in proto %d" % len(foo))

        num += 1
        elapsed = time() - start
        print(num, num/elapsed, x.shape, y.shape, z.shape )

        if save:
            save_images(num, x, y, z)


def main(type, batch_size, save):

    client = DataGeneratorClient(port=5556, host="localhost", hwm=1, batch_size=batch_size)

    if type=='processed':
        time_processed(client, batch_size)
    elif type=='raw':
        time_raw(client, save)
    else:
        assert False, "type should be 'processed' or 'raw' "


assert len(sys.argv) >=2,  "Usage: ./rmpe_dataset_server_tester <processed|raw> [batch_size] [save]"
batch_size=1
save = False
if 'save' in sys.argv:
    save=True
    sys.argv = [s for s in sys.argv if s!='save']
if len(sys.argv)==3: batch_size=int(sys.argv[2])

np.set_printoptions(precision=1, linewidth=1000, suppress=True, threshold=100000)
main(sys.argv[1], batch_size, save)
