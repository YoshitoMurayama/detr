# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd

from util.misc import all_gather


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.anns = []

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            if self.anns == []:
                self.anns = list(coco_dt.anns.values())
            else:
                self.anns += list(coco_dt.anns.values())
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################


def get_AP(dtm, gtIg, dtIg, plot=False):
    npig = (~gtIg).sum()
    if npig == 0:
        return -1
    eps = np.spacing(1)
    tps = np.cumsum(np.logical_and(dtm > 0, ~dtIg))
    fps = np.cumsum(np.logical_and(dtm == 0, ~dtIg))
    precision = tps / (tps + fps + eps)
    recall = tps / npig

    if len(precision) > 100:
        idx = np.linspace(0, len(precision) - 1, 100).astype('int')
        precision = precision[idx]
        recall = recall[idx]

    q = np.zeros(len(precision))
    _max = 0
    for i in np.arange(len(precision))[::-1]:
        if _max < precision[i]:
            _max = precision[i]
        q[i] = _max
    q = np.r_[q[0], q, 0] if len(q) > 1 else np.r_[q, q, 0]
    r = recall
    r = np.r_[0, r, r[-1]] if len(q) > 1 else np.r_[0, r, r]
    if plot:
        _df = pd.DataFrame([q, r]).T
        _df.columns = ['q', 'r']
        pprint(_df)

    if plot:
        plt.figure(figsize=(12, 3))
        plt.plot(recall, precision, '-o', lw=3, alpha=0.6)
        plt.fill_between(r, np.zeros(len(q)), q, fc='g', alpha=0.3)
        plt.grid()
        plt.ylim(0, 1.1)
        plt.xlim(0, 1.1)

    pr, idx = np.unique(q, return_index=True)
    idx = idx[np.argsort(idx)]
    pr = q[idx[1:] - 1]
    r = r[idx]
    r = r[1:] - r[:-1]
    if plot:
        _df = pd.DataFrame([pr, r]).T
        _df.columns = ['pr', 'r']
        pprint(_df)
    return np.dot(pr, r)


def get_ImageAP(evalImgs, image_id, plot=False, th=None, Ti=0):
    E = [e for e in evalImgs if e['image_id'] == image_id]
    if len(E) == 0:
        return -1
    dtScore = np.hstack([e['dtScores'] for e in E])
    sort_idx = np.argsort(-dtScore)  # if e['category_id']==1]))
    if th is not None and th != 0:
        sort_idx = sort_idx[:np.argmax(dtScore[sort_idx] < th)]
        if len(sort_idx) == 0:
            return -1
    dtm = np.hstack([e['dtMatches'][Ti] for e in E])[sort_idx]
    gtIg = np.hstack([e['gtIgnore'] for e in E]).astype('?')
    dtIg = np.hstack([e['dtIgnore'][Ti] for e in E])[sort_idx]
    return get_AP(dtm, gtIg, dtIg, plot=plot)


def get_CategoryAP(evalImgs, c, plot=False, th=None, Ti=0):
    E = [e for e in evalImgs if e['category_id'] == c]
    if len(E) == 0:
        return -1
    dtScore = np.hstack([e['dtScores'] for e in E])
    sort_idx = np.argsort(-dtScore)  # if e['category_id']==1]))
    if th is not None and th != 0:
        sort_idx = sort_idx[:np.argmax(dtScore[sort_idx] < th)]
        if len(sort_idx) == 0:
            return -1
    dtm = np.hstack([e['dtMatches'][Ti] for e in E])[sort_idx]
    gtIg = np.hstack([e['gtIgnore'] for e in E]).astype('?')
    dtIg = np.hstack([e['dtIgnore'][Ti] for e in E])[sort_idx]
    return get_AP(dtm, gtIg, dtIg, plot=plot)


def get_AreaAP(evalImgs, anns, minTh, maxTh, plot=False, Ti=0):
    E = evalImgs
    if len(E) == 0:
        return -1

    def _get(e):
        gtIds = np.array(e['gtIds'])
        areas = np.array([anns[i]['area'] for i in gtIds])
        mask = np.logical_and(areas >= minTh, areas < maxTh)
        _gtIds = gtIds[~mask]
        dtIg = np.array(e['dtIgnore'][Ti])
        dtM = e['dtMatches'][Ti]
        dtIg = np.array([np.logical_or(x, y in _gtIds) for x, y in zip(dtIg, dtM)], dtype='?')
        gtIg = np.array(e['gtIgnore'], dtype='?')
        gtIg[~mask] = True
        return gtIg, dtIg

    tmp = [_get(e) for e in E]
    dtScore = np.hstack([e['dtScores'] for e in E])
    sort_idx = np.argsort(-dtScore)
    gtIg = np.hstack([e[0] for e in tmp])
    dtIg = np.hstack([e[1] for e in tmp])[sort_idx]
    dtm = np.hstack([e['dtMatches'][Ti] for e in E])[sort_idx]
    return get_AP(dtm, gtIg, dtIg, plot=plot)


def get_RatioAP(evalImgs, anns, minTh, maxTh, plot=False, Ti=0):
    E = evalImgs
    if len(E) == 0:
        return -1

    def _get(e):
        gtIds = np.array(e['gtIds'])
        ratios = np.array([np.log(anns[i]['bbox'][2] / anns[i]['bbox'][3]) for i in gtIds])
        mask = np.logical_and(ratios >= minTh, ratios < maxTh)
        _gtIds = gtIds[~mask]
        dtIg = np.array(e['dtIgnore'][Ti])
        dtM = e['dtMatches'][Ti]
        dtIg = np.array([np.logical_or(x, y in _gtIds) for x, y in zip(dtIg, dtM)], dtype='?')
        gtIg = np.array(e['gtIgnore'], dtype='?')
        gtIg[~mask] = True
        return gtIg, dtIg

    tmp = [_get(e) for e in E]
    dtScore = np.hstack([e['dtScores'] for e in E])
    sort_idx = np.argsort(-dtScore)
    gtIg = np.hstack([e[0] for e in tmp])
    dtIg = np.hstack([e[1] for e in tmp])[sort_idx]
    dtm = np.hstack([e['dtMatches'][Ti] for e in E])[sort_idx]
    return get_AP(dtm, gtIg, dtIg, plot=plot)


def get_ARAP(evalImgs, anns, areaTh, ratioTh, plot=False, Ti=0, return_categories=False):
    area_min, area_max = areaTh
    ratio_min, ratio_max = ratioTh
    E = evalImgs
    if len(E) == 0:
        return -1

    def _get(e):
        gtIds = np.array(e['gtIds'])
        areas = np.array([anns[i]['area'] for i in gtIds])
        ratios = np.array([np.log(anns[i]['bbox'][2] / anns[i]['bbox'][3]) for i in gtIds])
        mask = np.logical_and(
            np.logical_and(areas >= area_min, ratios < area_max),
            np.logical_and(ratios >= ratio_min, ratios < ratio_max),
        )
        _gtIds = gtIds[~mask]
        dtIg = np.array(e['dtIgnore'][Ti])
        dtcats = np.ones(len(dtIg))*e['category_id']
        gtcats = np.ones(len(gtIds))*e['category_id']
        dtM = e['dtMatches'][Ti]
        dtIg = np.array([np.logical_or(x, y in _gtIds) for x, y in zip(dtIg, dtM)], dtype='?')
        gtIg = np.array(e['gtIgnore'], dtype='?')
        gtIg[~mask] = True
        return gtIg, dtIg, gtcats, dtcats

    tmp = [_get(e) for e in E]
    dtScore = np.hstack([e['dtScores'] for e in E])
    sort_idx = np.argsort(-dtScore)
    gtIg = np.hstack([e[0] for e in tmp])
    dtIg = np.hstack([e[1] for e in tmp])[sort_idx]
    dtm = np.hstack([e['dtMatches'][Ti] for e in E])[sort_idx]
    if return_categories:
        gtcats = np.hstack([e[2] for e in tmp])
        cats = np.hstack([e[3] for e in tmp])[sort_idx]
        cats = {c:[np.where(cats==c)[0], np.where(gtcats==c)[0]] for c in np.unique(cats)}
        cats = {c : get_AP(dtm[idx[0]], gtIg[idx[1]], dtIg[idx[0]]) for c, idx in cats.items()}
        return get_AP(dtm, gtIg, dtIg, plot=plot), cats

    return get_AP(dtm, gtIg, dtIg, plot=plot)