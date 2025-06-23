# C3: Creative Concept Catalyst

Jiyeon Han* · Dahee Kwon* · Gayoung Lee · Junho Kim · Jaesik Choi (* Equal Contribution)  

This is the official implementation of **Enhancing Creative Generation on Stable Diffusion-based Models**, published in CVPR 2025.

## Abstract
Recent text-to-image generative models, particularly Stable Diffusion and its distilled variants, have achieved impressive fidelity and strong text-image alignment. However, their creative capability remains constrained, as including `creative' in prompts seldom yields the desired results. In this paper, we introduce C3 (Creative Concept Catalyst), a training-free approach designed to enhance creativity in Stable Diffusion-based models. C3 selectively amplifies features during the denoising process to foster more creative outputs. We offer practical guidelines for choosing amplification factors based on two main aspects of creativity. C3 is the first study to enhance creativity in diffusion models without extensive computational costs. 

![image](./C3-main.png)

## Setup
1) To create the conda environment needed to run the code, run the following command:

```
conda env create -f environment.yaml
conda activate C3
```

2) You need to use CLIP code from "https://github.com/openai/CLIP":

```
git clone https://github.com/openai/CLIP
cd CLIP
pip install -e .
```

3) To use our source code, you also need to install diffusers from source "https://github.com/huggingface/diffusers.git":

```
git clone --branch v0.31.0 https://github.com/huggingface/diffusers.git
```

4) Please change the source code for diffusers below to the code for diffusers_custom and then install from source:
- diffusers/pipelines/stable_diffusion_xl/pipeline_stasble_diffuson_xl.py
- diffusers/models/unets/unet_2d_condition.py

```
cp diffusers_custom/pipeline_stable_diffusion_xl.py diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
cp diffusers_custom/unet_2d_condition.py diffusers/src/diffusers/models/unets/unet_2d_condition.py
cd diffusers
pip install -e .
```  

5) (Optional) If you use older GPU drivers (i.e., CUDA 11.8) please try this version of pytorch.
 
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```  

Now, you are ready to play with C3. 

 ## Usage
 1) You can try our method by manually setting the amplification factors in a simple ipython notebook `image_generation_examples.ipynb`.

 2) Or you can automatically find the amplification factors with the following sample script. 

```
bash scripts/amplification_factors.sh
```
To change the configuration, you can directly modify `src/default.cfg` or use `src/configuration.ipynb` to generate configuration automatically.
