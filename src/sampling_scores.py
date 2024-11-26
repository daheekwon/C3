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
    parser.add_argument("--obj", default="chair", type=str, help="A creative 'object'")
    parser.add_argument("--start_seed", type=int, default=0, help="starting seed")
    parser.add_argument("--n_samples", type=int, default=10, help="n_samples")
    parser.add_argument("--model", default="sdxl-turbo",type=str, help="Backbone models: sdxl-turbo or sdxl-light-1")
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

        
def compute_score(cmodel,c_preprocess,amodel,a_preprocessor,obj,seeds,model_name):
    # Define the path to the directory containing the image samples
    clip_ls_all = []
    aes_ls_all = []
    for seed in seeds:
        image_folder = f"/results/{model_name}/{obj}/seed_{seed}"

        # Create four lists for each category (0 to 3)
        img_lists = [{} for _ in range(4)]

        # Loop through each file in the folder
        for filename in os.listdir(image_folder):
            if filename.endswith(".png"):  # Check if the file is a PNG image
                # Split the filename to extract number1 and number2
                parts = filename.split('_')
                
                if len(parts) == 3:  # Ensure the filename format matches
                    number1 = int(parts[1])  # Convert to integer
                    number2 = int(parts[2].split('.')[0])*0.01  # Remove the file extension and convert to integer
                    

                    # Load the image
                    image_path = os.path.join(image_folder, filename)
                    image = Image.open(image_path)
                    
                    # Add to the corresponding category list
                    img_lists[number1][number2] = image
                    
                    org_path = f"/results/{model_name}/{obj}/seed_{seed}/sample_org.png"
                    org_img = Image.open(org_path)
                    img_lists[number1][0] = org_img
                    img_lists[number1] = dict(sorted(img_lists[number1].items()))

        txt = f'A creative {obj}'

        clip_ls = []
        aes_ls = []
        for list_img in img_lists:

            clip_ls_ = []
            aes_ls_ = []
            for amp, img in list_img.items():
                # preprocess image
                pixel_values = (
                    a_preprocessor(images=img.convert("RGB"), return_tensors="pt")
                    .pixel_values.to(torch.bfloat16)
                    .cuda()
                )

                with torch.no_grad(), torch.inference_mode():
                    aesthetic_score = amodel(pixel_values).logits.squeeze().float().cpu().numpy() 
                
                device = amodel.device
                text = clip.tokenize(txt).to(device)
                text_features = cmodel.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                image = c_preprocess(img).unsqueeze(0).to(device)
                image_features = cmodel.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                clip_score = (image_features @ text_features.T).detach().cpu().numpy().flatten()[0] 

                clip_ls_.append(float(clip_score))
                aes_ls_.append(float(aesthetic_score))
                del text, text_features, image, image_features, clip_score, aesthetic_score, pixel_values

        
            clip_ls.append(clip_ls_)
            aes_ls.append(aes_ls_)
        clip_ls_all.append(clip_ls)
        aes_ls_all.append(aes_ls)
        del clip_ls, aes_ls, clip_ls_, aes_ls_, img_lists

    
    return clip_ls_all, aes_ls_all
    

def main():
    args = get_args()
    start_seed = args.start_seed 
    obj = args.obj
    model_name = args.model
    n_samples = args.n_samples
    
    
    cmodel, c_preprocess = clip.load("ViT-B/32", device="cuda")

    # load model and preprocessor

    amodel, a_preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    amodel = amodel.to(torch.bfloat16).cuda()
    
    seeds = np.arange(start_seed,start_seed+n_samples,1)

    clip_ls_all, aes_ls_all = compute_score(cmodel,c_preprocess,amodel,a_preprocessor,obj,seeds,model_name)
    print(f"save {obj} usability score in seed {start_seed} to {start_seed+n_samples}...")
    
    filename_clip = f"/results/{model_name}/{obj}/clip_score.json"
    clip_current_list = load_list(filename_clip)
    clip_current_list.append(clip_ls_all)
    save_list(clip_current_list, filename_clip)
    
    filename_aes = f"/results/{model_name}/{obj}/aes_score.json"
    aes_current_list = load_list(filename_aes)
    aes_current_list.append(aes_ls_all)
    save_list(aes_current_list, filename_aes)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

