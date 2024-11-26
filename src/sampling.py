import os
import sys
import torch
import random
import numpy as np
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline, AutoPipelineForText2Image, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from matplotlib import gridspec
from matplotlib import pyplot as plt
from score_util_pub import *
import argparse

torch.cuda.empty_cache()

# seed, prompt, mask_attribute, nth att, name
def get_args():
    parser = argparse.ArgumentParser(description="Arguments for C3 methods",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obj", default="chair", type=str, help="A creative 'object'")
    parser.add_argument("--n_samples", type=int, default=100, help="n_samples")
    parser.add_argument("--model", default="sdxl-turbo",type=str, help="Backbone models: sdxl-turbo or sdxl-light-1")
    
    return parser.parse_args()

def sampling(model, obj, seed, model_name):
    tar_prompt = f"a creative {obj}"

    range1 = np.arange(1.1, 2.01, 0.1).tolist()
    range2 = np.arange(2.0, 5.01, 0.25).tolist()
    amp_factor = {0:sorted(set(range1 + range2)),1:sorted(set(range1 + range2)),2:list(np.arange(2,10.01,1)),3:list(np.arange(2,10.01,1))}
    
    set_seed(int(seed))
    output_dir = f"/results/{model_name}/{obj}/seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    out = model(tar_prompt,guidance_scale=0,num_inference_steps=1)
    img = out[0].images[0]
    img.save(f"/results/{model_name}/{obj}/seed_{seed}/sample_org.png")
    
    cutoffs = fold_mask([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    empty_mask = [[] for i in range(7)]
    empty_mask = fold_mask(empty_mask)
    dummy_mask = {t: empty_mask for t in range(1)}

    for key,value in amp_factor.items():
        for val in value:
            amp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            amp[key] = val
            replace_mask = fold_mask(amp)
            set_seed(int(seed))
            out = model(
            prompt=tar_prompt, num_inference_steps=1, guidance_scale=0, replace_mask=replace_mask, replace_on='freq', hidden_mask=dummy_mask, cutoff_freq=cutoffs
            )
            img = out[0].images[0]
            img.save(f"/results/{model_name}/{obj}/seed_{seed}/sample_{key}_{int(val*100)}.png")
    

def main():
    args = get_args()
    
    obj = args.obj
    n_samples = args.n_samples
    model_name = args.model
    
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
        sampling(pipe, obj, seed, model_name)
        print(f"Sampling in {seed}-seed is done.")

    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()