#!/bin/bash
#SBATCH --job-name=feature_map
#SBATCH --partition=gpu 
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=100g
#SBATCH --output=log_%j.log 
#SBATCH --error=error_%j.err 
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=tt920@lms.mrc.ac.uk

module load miniconda3
conda activate hisinet_test
cd /home/tt920/HiTwinFormer

echo "start extraction"

python make_feature_map.py \
    --model_name SLeNet \
    --model_file model_outputs/ctcf_final/SLeNet_CTCF_final_0.0001_32_contrastive_30004_Adam_0_aug_.ckpt \
    --save_path feature_maps/ctcf_degron_bs32_contrastive_0_aug_LeNet_no_saliency.fmlhic \
    --reference_genome mm10 \
    --json_file D4_CTCF_224.json

python make_feature_map.py \
    --model_name SLeNet \
    --model_file model_outputs/ctcf_final/SLeNet_CTCF_final_0.0001_32_contrastive_30004_Adam_0_aug_.ckpt \
    --save_path feature_maps/ctcf_degron_bs32_contrastive_0_aug_LeNet_saliency.fmlhic \
    --reference_genome mm10 \
    --json_file D4_CTCF_224.json \
    --saliency_mapping

python -u make_feature_map.py \
    --model_name SMaxVit \
    --model_file model_outputs/ctcf_final/maxvit/contrastive_triplet/SMaxVit_CTCF_final_0.0001_32_contrastive_30004_Adam_0_aug_final.ckpt \
    --save_path feature_maps/ctcf_degron_bs32_contrastive_0_aug_MaxVit_saliency.fmlhic \
    --reference_genome mm10 \
    --json_file D4_CTCF_224.json \
    --saliency_mapping

conda deactivate