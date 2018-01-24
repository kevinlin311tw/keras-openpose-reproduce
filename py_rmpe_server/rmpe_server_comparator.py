#!/usr/bin/env python

import sys
import os
sys.path.append("..")

from time import time
from training.ds_generators import DataGeneratorClient
from py_rmpe_config import RmpeGlobalConfig

import numpy as np
import pandas as pd
import cv2

servers = [('py-server', 'localhost', 5556), ('cpp-server', 'localhost', 5557)]
clients = {}
save_to = 'comparator'  # save new server to output, c++ server to original and compare images

def cmp_pics(num, lhsd, rhsd, lhsn, rhsn):
    diff = lhsd.astype(float) - rhsd.astype(float)
    L1 = np.average(np.abs(diff))
    L2 = np.sqrt(np.average(diff**2))
    AC = np.average(lhsd==rhsd)

    print("Image: ", num, lhsd.shape, rhsd.shape, L1, L2, AC)

    diff = diff.transpose([1,2,0])
    diff = np.abs(diff)
    diff = diff.astype(np.uint8)

    cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "image." + lhsn), lhsd.transpose([1,2,0]))
    cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "image." + rhsn), rhsd.transpose([1,2,0]))
    cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "imagediff" ), diff)

    return (L1,L2,AC)

def cmp_masks(num, lhsd, rhsd, lhsn, rhsn):
    diff = (lhsd.astype(float) - rhsd.astype(float))*255.0
    L1 = np.average(np.abs(diff))
    L2 = np.sqrt(np.average(diff**2))
    AC = np.average(lhsd == rhsd)

    print("Mask: ", num, lhsd.shape, rhsd.shape, L1, L2, AC)

    lhsd = lhsd.reshape((lhsd.shape[0], lhsd.shape[1], 1))
    lhsd = (lhsd*255).astype(np.uint8)
    lhsd = cv2.resize(lhsd, (RmpeGlobalConfig.height, RmpeGlobalConfig.width), interpolation=cv2.INTER_NEAREST)
    lhsd = lhsd.reshape((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 1))

    rhsd = rhsd.reshape((rhsd.shape[0], rhsd.shape[1], 1))
    rhsd = (rhsd*255).astype(np.uint8)
    rhsd = cv2.resize(rhsd, (RmpeGlobalConfig.height, RmpeGlobalConfig.width), interpolation=cv2.INTER_NEAREST)
    rhsd = rhsd.reshape((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 1))

    diff = np.abs(diff).reshape((diff.shape[0], diff.shape[1], 1))
    diff = diff.astype(np.uint8)
    diff = cv2.resize(diff, (RmpeGlobalConfig.height, RmpeGlobalConfig.width), interpolation=cv2.INTER_NEAREST)
    diff = diff.reshape((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 1))

    cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "mask."+lhsn), lhsd)
    cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "mask."+rhsn), rhsd)
    cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "maskdiff"), diff)

    return (L1,L2,AC)

def cmp_layers(num, lhsd_all, rhsd_all, lhsn, rhsn):

    result = []

    L1T = 0
    L2T = 0
    ACT = 0

    for layer in range(RmpeGlobalConfig.num_layers):
        lhsd = lhsd_all[layer, :, :]
        rhsd = rhsd_all[layer, :, :]

        diff = (lhsd.astype(float) - rhsd.astype(float))*255.0
        L1 = np.average(np.abs(diff))
        L2 = np.sqrt(np.average(diff**2))
        AC = np.average(lhsd == rhsd)

        #print("Layers(%d): " % layer, num, lhsd.shape, rhsd.shape, L1, L2, AC)
        L1T += L1
        L2T += L2
        ACT += AC

        lhsd = lhsd.reshape((lhsd.shape[0], lhsd.shape[1], 1))
        lhsd = (127+lhsd*128).astype(np.uint8)
        lhsd = cv2.resize(lhsd, (RmpeGlobalConfig.height, RmpeGlobalConfig.width), interpolation=cv2.INTER_NEAREST)
        lhsd = lhsd.reshape((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 1))

        rhsd = rhsd.reshape((rhsd.shape[0], rhsd.shape[1], 1))
        rhsd = (127+rhsd*128).astype(np.uint8)
        rhsd = cv2.resize(rhsd, (RmpeGlobalConfig.height, RmpeGlobalConfig.width), interpolation=cv2.INTER_NEAREST)
        rhsd = rhsd.reshape((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 1))

        diff = np.abs(diff).reshape((diff.shape[0], diff.shape[1], 1))
        diff = diff.astype(np.uint8)
        diff = cv2.resize(diff, (RmpeGlobalConfig.height, RmpeGlobalConfig.width), interpolation=cv2.INTER_NEAREST)
        diff = diff.reshape((RmpeGlobalConfig.height, RmpeGlobalConfig.width, 1))

        cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "layer" + str(layer) + "." + lhsn), lhsd)
        cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "layer" + str(layer) + "." + rhsn), rhsd)
        cv2.imwrite(save_to+("/%5d" % num)+"/%07d%s.png" % (num, "layer" + str(layer) + "diff"), diff)

        result += [L1, L2, AC]

    print("Layers: ", num, lhsd.shape, rhsd.shape, L1T/RmpeGlobalConfig.num_layers, L2T/RmpeGlobalConfig.num_layers, ACT/RmpeGlobalConfig.num_layers)

    return result

def step(num, augs):

    all_res = []

    os.makedirs(save_to+("/%5d" % num), exist_ok=True)

    for (i,lhs) in enumerate(augs):
        for (j,rhs) in enumerate(augs):
            if i < j:

                res = []

                res += cmp_pics(num, augs[lhs][0], augs[rhs][0], lhs, rhs)
                res += cmp_masks(num, augs[lhs][1], augs[rhs][1], lhs, rhs)
                res += cmp_layers(num, augs[lhs][2], augs[rhs][2], lhs, rhs)

                all_res += [res]

    return all_res

def main(servers, batch_size):

    for (name, host, port) in servers:
        clients[name] = DataGeneratorClient(port=port, host=host, hwm=1, batch_size=batch_size).gen_raw()

    res_all = []

    for i in range(2645): #2645
        print(i)
        augs = dict([(name, next(value)) for (name, value) in clients.items()])
        res = step(i, augs)
        res_all += res

    columns = ["ImageL1", "ImageL2", "ImageAC", "MaskL1", "MaskL2", "MaskAC"]
    for layer in range(RmpeGlobalConfig.num_layers):
        columns += ["Layer"+str(layer)+"L1", "Layer"+str(layer)+"L2", "Layer"+str(layer)+"AC"]

    res_all = np.array(res_all)
    print(res_all.shape)

    results = pd.DataFrame(res_all, columns=columns )
    results.to_csv("weights.tsv", sep="\t")

batch_size=20
np.set_printoptions(precision=1, linewidth=1000, suppress=True, threshold=100000)
main(servers, batch_size)
