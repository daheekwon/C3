# C3: Creatvie Concept Catalyst

## Description
Official implementation of our C3 paper.

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