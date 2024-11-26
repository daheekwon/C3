CUDA_VISIBLE_DEVICES=1


obj="chair"
thres=80
model="sdxl-turbo" #For SDXL Lightning 1-step, use "sdxl-light-1"

### 1) Sampling Imges
python src/sampling.py --obj "$obj" --n_samples 100 --model "$model"

### 2) Computing the Usability Score
start_seed=0
end_seed=9
for ((seed=$start_seed; seed<=$end_seed; seed++))
do
    seed_value=$((seed * 10))
    python src/sampling_scores.py --start_seed $seed_value --obj "$obj" --model "$model"
done 

### 3) Extracting Optimal Amplification Factors with usability
python src/sampling_final_scores.py --obj "$obj" --thres "$thres" --model "$model"


