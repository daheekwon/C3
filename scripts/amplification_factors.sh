# Specify cuda device here
# CUDA_VISIBLE_DEVICES=1


# obj="car"
# thres=80
# model="sdxl-turbo" #For SDXL Lightning 1-step, use "sdxl-light-1"

### 1) Sampling Imges
python src/sampling.py --conf_file "src/default.cfg" #--obj "$obj" --n_samples 100 --model "$model"

### 2) Computing the Usability Score
start_seed=0
end_seed=0
for ((seed=$start_seed; seed<=$end_seed; seed++))
do
    seed_value=$((seed * 10))
    python src/sampling_scores.py --conf_file "src/default.cfg" --start_seed $seed_value # --model "$model"
done 

### 3) Extracting Optimal Amplification Factors with usability
python src/sampling_final_scores.py --conf_file "src/default.cfg" #--obj "$obj" --thres "$thres" --model "$model"


