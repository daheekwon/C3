import os
from score_util_pub import *
import random
import torch
from diffusers import DDIMScheduler, DiffusionPipeline, PixArtAlphaPipeline, HunyuanDiTPipeline, StableDiffusion3Pipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, Kandinsky3Pipeline 
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class Generator():
    def __init__(self, gen_name):
        self.model = None
        self.model_name = gen_name
        if gen_name == 'sdxl':
            self.model = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.model.to("cuda")

        elif gen_name == 'sdxl-turbo':
            self.model = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
            self.model.to("cuda")
        
        elif gen_name == 'sdxl-light-1':            
            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"
            # ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
            # predict_type = "epsilon"
            ckpt = "sdxl_lightning_1step_unet_x0.safetensors" # Use the correct ckpt for your step setting!
            predict_type = "sample"

            # Load model.
            unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
            self.model = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
            self.model.scheduler = EulerDiscreteScheduler.from_config(self.model.scheduler.config, timestep_spacing="trailing", prediction_type=predict_type)

        elif gen_name == 'sdxl-light-4':            
            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"
            ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
            predict_type = "epsilon"
            # ckpt = "sdxl_lightning_1step_unet_x0.safetensors" # Use the correct ckpt for your step setting!
            # predict_type = "sample"

            # Load model.
            unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
            self.model = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
            self.model.scheduler = EulerDiscreteScheduler.from_config(self.model.scheduler.config, timestep_spacing="trailing", prediction_type=predict_type)
        

    def orig(self, prompt, seed=None, save_path=''):
        if not isinstance(seed, int):
            seed = random.randint(0,10)
        set_seed(seed)
        params = {'prompt': prompt}
        if 'turbo' in self.model_name or 'light' in self.model_name:
            params['guidance_scale'] = 0.0
        if 'turbo' in self.model_name or 'light-1' in self.model_name:
            params['num_inference_steps'] = 1
        elif 'light-4' in self.model_name:
            params['num_inference_steps'] = 4

        image = self.model(**params)[0].images[0]
        if save_path != '':
            image.save(save_path)

        return image

    def c3(self, seed, prompt, replace_mask, save_path='', cutoff=5.0):
        set_seed(seed)
        if 'turbo' in self.model_name or 'light-1' in self.model_name:
            c3_step = 1
        elif 'light-4' in self.model_name:
            c3_step = 4
        elif self.model_name == 'sdxl':
            c3_step = 50
        empty_mask = [[] for i in range(7)]
        empty_mask = fold_mask(empty_mask)
        dummy_mask = {t: empty_mask for t in range(c3_step)}
        # prompt = "a creative chair"
        if isinstance(cutoff, float):
            cutoff = [cutoff]*4 + [1.0]*3

        cutoff = fold_mask(cutoff)

        replace_mask_ = fold_mask(replace_mask)

        params = {'prompt': prompt, 'replace_mask':replace_mask_, 'hidden_mask': dummy_mask, 'replace_on':'freq', 'cutoff_freq': cutoff}
        if 'turbo' in self.model_name or 'light' in self.model_name:
            params['guidance_scale'] = 0.0
        if 'turbo' in self.model_name or 'light-1' in self.model_name:
            params['num_inference_steps'] = 1
        elif 'light-4' in self.model_name:
            params['num_inference_steps'] = 4

        out = self.model(
            **params
        )

        image = out[0].images[0]
        if save_path != '':
            image.save(save_path)
        
        return image