import os
import sys
import random
import torch
import numpy as np
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline, AutoPipelineForText2Image, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from matplotlib import gridspec
from matplotlib import pyplot as plt
from score_util_pub import *
import argparse
from PIL import Image
from glob import glob
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import gc
sys.path.insert(0, '/dir_for_CLIPS/CLIP/')
import clip
import argparse
import json

torch.cuda.empty_cache()


# seed, prompt, mask_attribute, nth att, name
def get_args():
    parser = argparse.ArgumentParser(description="Arguments for C3 methods",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--conf_file", default="src/default.cfg", type=str, help="configuration file")
    # parser.add_argument("--obj", default="chair", type=str, help="A creative 'object'")
    # parser.add_argument("--thres", type=int, default=80, help="thrshold to choose amplifying factor")
    # parser.add_argument("--model", default="sdxl-turbo",type=str, help="Backbone models: sdxl-turbo or sdxl-light-1")
    return parser.parse_args()


# Function to load the list from the file
def load_list(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        return []  # Return an empty list if the file doesn't exist

# Function to save the list to the file
def save_list(data_list, filename):
    with open(filename, 'w') as file:
        json.dump(data_list, file)
        

def compute_amp_factor(clip_ls_all,aes_ls_all,thres):
    
    range1 = np.arange(1.1, 2.01, 0.1).tolist()
    range2 = np.arange(2.0, 5.01, 0.25).tolist()
    amp_factor = {0:sorted(set(range1 + range2)),1:sorted(set(range1 + range2)),2:list(np.arange(2,10.01,1)),3:list(np.arange(2,10.01,1))}
    
    clip_0 = np.mean([clip_ls_all[i][j][0] for j in range(len(clip_ls_all[0])) for i in range(len(clip_ls_all))], axis=0)
    clip_1 = np.mean([clip_ls_all[i][j][1] for j in range(len(clip_ls_all[0])) for i in range(len(clip_ls_all))], axis=0)
    clip_2 = np.mean([clip_ls_all[i][j][2] for j in range(len(clip_ls_all[0])) for i in range(len(clip_ls_all))], axis=0)
    clip_3 = np.mean([clip_ls_all[i][j][3] for j in range(len(clip_ls_all[0])) for i in range(len(clip_ls_all))], axis=0)

    aes_0 = np.mean([aes_ls_all[i][j][0] for j in range(len(aes_ls_all[0])) for i in range(len(aes_ls_all))], axis=0)
    aes_1 = np.mean([aes_ls_all[i][j][1] for j in range(len(aes_ls_all[0])) for i in range(len(aes_ls_all))], axis=0)
    aes_2 = np.mean([aes_ls_all[i][j][2] for j in range(len(aes_ls_all[0])) for i in range(len(aes_ls_all))], axis=0)
    aes_3 = np.mean([aes_ls_all[i][j][3] for j in range(len(aes_ls_all[0])) for i in range(len(aes_ls_all))], axis=0)

    clip_ls = [clip_0, clip_1, clip_2, clip_3]
    aes_ls = [aes_0, aes_1, aes_2, aes_3]
    
    aes_min = min([np.min(aes_ls[i]) for i in range(len(aes_ls))])
    aes_max = max([np.max(aes_ls[i]) for i in range(len(aes_ls))])

    clip_min = min([np.min(clip_ls[i]) for i in range(len(clip_ls))])
    clip_max = max([np.max(clip_ls[i]) for i in range(len(clip_ls))])
    
    aes_ls = [(aes_ls[i] - aes_min) / (aes_max - aes_min) for i in range(len(aes_ls))]
    clip_ls = [(clip_ls[i] - clip_min) / (clip_max - clip_min) for i in range(len(clip_ls))]

    final_score = [aes_ls[i] + clip_ls[i] for i in range(len(aes_ls))]
    
    del aes_ls, clip_ls, aes_ls_all, clip_ls_all
    
    optimal_amps = []
    for l in range(4):
        optimal_amp = 1
        for i,amp in enumerate(amp_factor[l]):
            if (i != 0) and final_score[l][i] >= final_score[l][0]*thres:
                optimal_amp = max(optimal_amp, amp)
        optimal_amps.append(optimal_amp)
    return optimal_amps

    

def main():
    args = get_args()
    conf_file = args.conf_file
    with open(conf_file, 'r') as conf:
        config = json.load(conf)
    obj = config["obj"]
    thres = config["use_thres"]#*0.01
    model_name = config["model"]
    base_dir = config["work_dir_prefix"]
        
    filename_clip = os.path.join(base_dir, f"{model_name}/{obj}/clip_score.json")
    clip_ls_all = load_list(filename_clip)
    filename_aes = os.path.join(base_dir, f"{model_name}/{obj}/aes_score.json")
    aes_ls_all = load_list(filename_aes)
    
    
    amp_factor = compute_amp_factor(clip_ls_all,aes_ls_all,float(thres)*0.01)
    
    # File to save and load the list
    print(f"save {obj} amplifying factors...")
    
    filename = os.path.join(base_dir, f"{model_name}/{obj}/amp_factors_{thres}.json")
    current_list = load_list(filename)
    current_list.append(amp_factor)
    save_list(current_list, filename)
    print(f"Updated list saved: {current_list}")
        

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

