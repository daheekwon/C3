import os
# import sys
import torch
# import random
import numpy as np
import json
# import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline, AutoPipelineForText2Image, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
from score_util_pub import *
import argparse

torch.cuda.empty_cache()

# seed, prompt, mask_attribute, nth att, name
def get_args():
    parser = argparse.ArgumentParser(description="Arguments for C3 methods",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--conf_file", default="src/default.cfg", type=str, help="configuration file")
   
    return parser.parse_args()

def sampling(model, tar_prompt, seed, save_dir, cutoffs, amp_factor, c3_steps, n_steps=-1, guidance_scale=-1):
    os.makedirs(save_dir, exist_ok=True)

    params = {'prompt': tar_prompt}
    if guidance_scale != -1:
        params['guidance_scale'] = guidance_scale
    if n_steps != -1:
        params['num_inference_steps'] = n_steps

    fname = os.path.join(save_dir, "sample_org.png")
    if not os.path.exists(fname): 
        set_seed(int(seed))
        out = model(**params)
        img = out[0].images[0]
        img.save(fname)
    
    cutoffs = fold_mask(cutoffs)
    
    empty_mask = [[] for i in range(7)]
    empty_mask = fold_mask(empty_mask)
    dummy_mask = {t: empty_mask for t in range(c3_steps)}
    params['hidden_mask'] = dummy_mask
    params['replace_on'] = 'freq'


    for key,value in amp_factor.items():
        for val in value:
            fname = os.path.join(save_dir, f"sample_{key}_{int(val*100)}.png")
            if os.path.exists(fname): continue
            amp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            amp[int(key)] = float(val)
            replace_mask = fold_mask(amp)
            params['replace_mask'] = replace_mask
            set_seed(int(seed))
            out = model(**params)
            img = out[0].images[0]
            img.save(fname)
    

def main():
    args = get_args()
    conf_file = args.conf_file
    with open(conf_file, 'r') as conf:
        config = json.load(conf)
    
    obj = config["obj"]
    n_samples = config["n_samples"]
    model_name = config["model"]
    base_dir = config["work_dir_prefix"]
    cutoff = config["cutoff"]
    amp_range = config["range"]
    prompt = config["prompt"].format(obj=obj)
    n_steps = config["n_steps"]
    c3_steps = config["c3_steps"]
    guidance_scale = config["guidance_scale"]

    if model_name == "sdxl-turbo":
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    elif model_name == "sdxl-light-1":
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_1step_unet_x0.safetensors" # For step 1
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", prediction_type="sample") # For step 1
    print("Model loaded.")
    
    for seed in range(n_samples):
        save_dir = os.path.join(base_dir, f'{model_name}/{obj}/seed_{seed}/')
        sampling(pipe, prompt, seed, save_dir, cutoff, amp_range, c3_steps, n_steps, guidance_scale)
        print(f"Sampling in {seed}-seed is done.")

    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()