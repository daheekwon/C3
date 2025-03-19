# C3: Creatvie Concept Catalyst

Jiyeon Han* · Dahee Kwon* · Gayoung Lee · Junho Kim · Jaesik Choi (* Equal Contribution)  

This is the official implementation of Enhancing Creative Generation on Stable Diffusion-based Models, published in CVPR 2025.

## Abstract
Recent text-to-image generative models, particularly Stable Diffusion and its distilled variants, have achieved impressive fidelity and strong text-image alignment. However, their creative capability remains constrained, as including `creative' in prompts seldom yields the desired results. In this paper, we introduce C3 (Creative Concept Catalyst), a training-free approach designed to enhance creativity in Stable Diffusion-based models. C3 selectively amplifies features during the denoising process to foster more creative outputs. We offer practical guidelines for choosing amplification factors based on two main aspects of creativity. C3 is the first study to enhance creativity in diffusion models without extensive computational costs. 


## Setup
1) To create the conda environment needed to run the code, run the following command:

```
conda env create -f environment.yaml
conda activate C3
```

2) You need to use CLIP code from "https://github.com/openai/CLIP":

```
git clone https://github.com/openai/CLIP
```

3) Please change the source code for diffusers below to the code for diffusers_custom:
- diffusers/pipelines/sdxl_diffusion_xl/pipeline_stasble_diffuson_xl.py
- diffusers/models/unets/unet_2d_condition.py



 ## Usage
 ```
bash scripts/amplification_factors.sh
```
