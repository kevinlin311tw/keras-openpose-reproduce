import sys
sys.path.append('../dataset/cocoapi/PythonAPI')
sys.path.append("..")
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import code
import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as io
import pylab
import os
import os.path
import pandas

# []
# orderCOCO = [1,0, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4]
orderCOCO = [0,1, 15,14,17,16, 5,2,6,3,7,4, 11,8,12,9,13,10]

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def load_cmu_val1k(mode):
    # mode == 1: load image id
    # mode == 2: load filename
    flist = []
    f = open('val2014_1k.txt','r')
    if mode==1:
        for line in f:
            flist.append(int(line.split()[mode]))
    else:        
        for line in f:
            flist.append(line.split()[mode])
    return flist

def get_last_epoch(training_log):
    print 'load training log: %s'%(training_log)
    data = pandas.read_csv(training_log)
    return max(data['epoch'].values)

def process_multi_scale (input_image, model, params, model_params):

    oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        # print scale
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)



    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)


    canvas = cv2.imread(input_image)  # B,G,R order
    
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    

    stickwidth = 4

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    

    return canvas, candidate, subset

def process_single_scale (input_image, model, params, model_params):

    oriImg = cv2.imread(input_image)  # B,G,R order

    heatmap_ori_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_ori_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    input_img = np.transpose(np.float32(oriImg[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

    conv_tic = time.time()
    output_blobs = model.predict(input_img)
    conv_toc = time.time()
    conv_cost = conv_toc - conv_tic

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    paf = np.squeeze(output_blobs[0])  # output 0 is PAFs

    heatmap_ori_size = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    paf_ori_size = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    post_proc_tic = time.time()
    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_ori_size[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_ori_size[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    post_proc_toc = time.time()
    post_proc_cost = post_proc_toc - post_proc_tic

    canvas = cv2.imread(input_image)  # B,G,R order
    
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    

    stickwidth = 4

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    

    return canvas, candidate, subset, conv_cost, post_proc_cost      


def compute_keypoints(model_weights_file, cocoGt, coco_api_dir, coco_data_type, eval_method, epoch_num):
    # load model
    model = get_testing_model()
    model.load_weights(model_weights_file)
    # load model config
    params, model_params = config_reader()

    # load epoch num
    trained_epoch = epoch_num
    # load validation image ids
    imgIds = sorted(cocoGt.getImgIds())

    # eval model
    mode_name = ''
    if eval_method==1:
        mode_name = 'open-pose-multi-scale'
    elif eval_method==0:
        mode_name = 'open-pose-single-scale'

    # prepare json output
    json_file = open(args.outputjson,'w')

    output_folder = './results/val2014-ours-epoch%d-%s'%(trained_epoch,mode_name)
    if not os.path.exists(output_folder):
	   os.mkdir(output_folder)

    prediction_folder = '%s/predictions'%(output_folder)
    if not os.path.exists(prediction_folder):
       os.mkdir(prediction_folder)

    # prepare json output
    json_fpath = '%s/%s'%(output_folder,args.outputjson)
    json_file = open(json_fpath,'w')


    candidate_set = []
    subset_set = []
    image_id_set = []
    counter = 0
    # run keypoints detection per image
    for item in imgIds:
        # load image fname
        fname = cocoGt.imgs[item]['file_name']
        input_fname = '%s/images/%s/%s'%(coco_api_dir,coco_data_type,fname)
        print input_fname
        print ('Image file exist? %s')%(os.path.isfile(input_fname)) 

        # run keypoint detection
        if eval_method==1:
            visual_result, candidate, subset= process_multi_scale(input_fname, model, params, model_params)
        elif eval_method==0:
            visual_result, candidate, subset, conv_cost, post_cost = process_single_scale(input_fname, model, params, model_params)

        # draw results       
        output_fname = '%s/result_%s'%(prediction_folder,fname)
        cv2.imwrite(output_fname, visual_result)
        candidate_set.append(candidate)
        subset_set.append(subset)
        image_id_set.append(item)  
        counter = counter + 1

    # dump results to json file
    write_json(candidate_set, subset_set, image_id_set, json_file)
    return json_fpath

def write_json(candidate_set, subset_set, image_id_set, json_file):
    category_id = 1
    output_data = []
    with json_file as outfile:
        total_images = len(subset_set)
        for i in range(total_images):
            valid_person_num = len(subset_set[i])
            for person in range(valid_person_num):
                valid_parts_num = subset_set[i][person][-1].astype(int)
                keypoints = []
                score = 0.0
                for part in range(18):
                    part_idx = orderCOCO[part]
                    if part_idx == 1:
                        # skip neck for coco eval
                        continue
                    else:
                        idx = subset_set[i][person][part_idx]
                        if idx.astype(int) == -1:
                            keypoints.append(0)
                            keypoints.append(0)
                            keypoints.append(0)
                        else:                  
                            x = candidate_set[i][idx.astype(int),0].astype(int)
                            y = candidate_set[i][idx.astype(int),1].astype(int)
                            # score = score + candidate_set[i][idx.astype(int),2]
                            keypoints.append(x)
                            keypoints.append(y)
                            keypoints.append(2)

                # score = score/valid_parts_num.astype(float)
                score = subset_set[i][person][-2]
                json_dict = {"image_id":image_id_set[i], "category_id":category_id, "keypoints": keypoints,"score":score}             
                output_data.append(json_dict)
        json.dump(output_data,outfile)
    # code.interact(local=locals()) 


def run_eval_metric(cocoGt, prediction_json, total_time, full_eval):
    #initialize COCO detections api
    annType = 'keypoints'
    cocoDt = cocoGt.loadRes(prediction_json)
    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    # load validation image ids
    imgIds = sorted(cocoGt.getImgIds())
    out_prefix = 'full'
    if full_eval==False:
	out_prefix = '1k'
        imgIds = load_cmu_val1k(mode=1)

    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
   
    # create output file for accuracy number
    scores = cocoEval.stats
    # serialize to file, to be read
    acc_file = ('%s_%s.txt')%(prediction_json,out_prefix)
    outputs = np.append(scores,total_time)
    np.savetxt(acc_file,outputs,delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    epoch_num = 100
    model_file = '../training/weights/weights.%04d.h5'%(epoch_num)
    parser.add_argument('--model', type=str, default=model_file, help='path to the weights file')
    parser.add_argument('--outputjson', type=str, default='val2014_result.json', help='path to the json file for coco eval')
    parser.add_argument('--coco_dataType', type=str, default='val2014', help='val2017 or val2014')
    parser.add_argument('--coco_api_dir', type=str, default='../dataset/cocoapi', help='path to coco api')
    parser.add_argument('--eval_method', type=int, default=0, help='open-pose-single-scale: 0, open-pose-multi-scale: 1')

    args = parser.parse_args()
    keras_weights_file = args.model
    # load coco eval api
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    annType = 'keypoints'
    prefix = 'person_keypoints'
    print 'COCO eval for *%s* results.'%(annType)

    #initialize COCO ground truth api
    annFile = '%s/annotations/%s_%s.json'%(args.coco_api_dir, prefix, args.coco_dataType)
    cocoGt = COCO(annFile)
    tic = time.time()
    print('start processing...')
    json_path = compute_keypoints(keras_weights_file, cocoGt, args.coco_api_dir, args.coco_dataType, args.eval_method, epoch_num)
    toc = time.time()
    total_time = toc - tic
    print ('overall processing time is %.5f' % (toc - tic))

    # run coco eval 2014
    run_eval_metric(cocoGt, json_path, total_time, full_eval=True)
    # run coco eval 2014 (1k images random selected by CMU)
    run_eval_metric(cocoGt, json_path, total_time, full_eval=False)


