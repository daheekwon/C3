import torch
import numpy as np
from collections import defaultdict
import cv2
import random


def convert_mask(mask):
    new_mask = defaultdict(list)
    n_layer = (len(mask)-1)//2
    for i in range(len(mask)):
        msk = mask[i]
        if i < n_layer: key = 'down'
        elif i == n_layer: key = 'mid'
        else: key = 'up'
        new_mask[key].append(np.array(msk))
    return new_mask


def mask_subtract(mask1, mask2):
    new_mask = defaultdict(list)
    for k in ['down', 'mid', 'up']:
        for l in range(len(mask1[k])):
            new_mask[k].append(np.setdiff1d(mask1[k][l],mask2[k][l]))
    return new_mask

def mask_intersect(mask1, mask2):
    new_mask = defaultdict(list)
    for k in ['down', 'mid', 'up']:
        for l in range(len(mask1[k])):
            new_mask[k].append(np.intersect1d(mask1[k][l],mask2[k][l]))
    return new_mask

def mask_union(mask1, mask2):
    if mask2 == None : return mask1
    new_mask = defaultdict(list)
    for k in ['down', 'mid', 'up']:
        for l in range(len(mask1[k])):
            new_mask[k].append(np.union1d(mask1[k][l],mask2[k][l]))
    return new_mask    

def mask_union_all(mask_list):
    mask_ = None
    for mask in mask_list:
        mask_ = mask_union(mask, mask_)
    return mask_



def fold_mask(mask):
    new_mask = defaultdict(list)
    n_layer = (len(mask)-1)//2
    for i in range(len(mask)):
        if i < n_layer: key = 'down'
        elif i == n_layer: key = 'mid'
        else: key = 'up'
        if i in range(len(mask)):
            new_mask[key].append(mask[i])
        else:
            new_mask[key].append([])

    return new_mask

def unfold_mask(mask, return_type='list'):
    new_mask = []
    for k in ['down','mid','up']:
        for l in mask[k]:
            new_mask.append(np.array(l).astype(int))
    if return_type=='list':
        return new_mask
    elif return_type=='dict':
        return {l: new_mask[l] for l in range(len(new_mask))}

def mask_to_mat(all_hidden_mask):
    all_hidden_index = []
    for t in range(50):
        tmp = mask_to_index(all_hidden_mask[t])
        all_hidden_index.append(tmp)
    all_hidden_index = np.array(all_hidden_index)
    return all_hidden_index

def mask_to_index(mask):
    tmp = []
    if 'down' in mask:
        mask = unfold_mask(mask, return_type='dict')
    for i in range(len(mask)):
        temp = np.zeros(lnums[len(mask)][i])
        if i in mask:
            temp[mask[i]] = 1
        tmp.extend(temp)
    return np.array(tmp)

def remove_layer(mask, layers):
    mask = unfold_mask(mask)
    for l in layers:
        mask[l] = []
    mask = fold_mask(mask)
    return mask

def get_mask_for_all_resolutions(fname, resolutions):
    mask = np.load(fname)[0]>0.5
    masks = []
    for res in resolutions:
        mask = cv2.resize(mask.astype(float), (res, res), interpolation = cv2.INTER_AREA).astype(bool)
        masks.append(mask)
    masks = fold_mask(masks)
    return masks

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)