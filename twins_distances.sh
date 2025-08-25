#!/bin/bash
#SBATCH --job-name=distance_twins
#SBATCH --partition=gpu 
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64g
#SBATCH --output=log_%j.log 
#SBATCH --error=error_%j.err 
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=tt920@lms.mrc.ac.uk

module load miniconda3
conda activate hisinet_test

cd /home/tt920/HiTwinFormer

# Run distance calculation with dynamic output CSV path

#python distance_calculation_all_stats_original_version_middle.py \
#  --mlhic_dataset_path "Paxip_224.json" \
#  --model_ckpt_path "model_outputs/paxip_final/SLeNet_Paxip_final_0.0001_64_triplet_30004_Adam_0_aug_final.ckpt" \
#  --output_dir "distances/Paxip/" \
#  --model_name "SLeNet" \
#  --reference_genome "mm9" \
#  --distance_measure "pairwise" \
#  --bin_size 10000 \
#  --patch_len 224 \
#  --position_stride 10 

#python distance_calculation_all_stats_original_version_middle.py \
#  --mlhic_dataset_path "Paxip_224.json" \
#  --model_ckpt_path "model_outputs/paxip_final/SLeNet_Paxip_final_0.0001_32_contrastive_30004_Adam_0_aug_final.ckpt" \
#  --output_dir "distances/Paxip/" \
#  --model_name "SLeNet" \
#  --reference_genome "mm9" \
#  --distance_measure "pairwise" \
#  --bin_size 10000 \
#  --patch_len 224 \
#  --position_stride 10 

#python distance_calculation_all_stats_original_version_middle.py \
#  --mlhic_dataset_path "Paxip_224.json" \
#  --model_ckpt_path "model_outputs/paxip_final/SLeNet_Paxip_final_0.0001_32_contrastive_30004_Adam_0_aug_final.ckpt" \
#  --output_dir "distances/Paxip/" \
#  --model_name "SLeNet" \
#  --reference_genome "mm9" \
#  --distance_measure "pairwise" \
#  --bin_size 10000 \
#  --patch_len 224 \
#  --position_stride 10 

#python distance_calculation_all_stats.py \
#  --mlhic_dataset_path "DP_SP_224.json" \
#  --model_ckpt_path "model_outputs/dp_ctcf/SLeNet_DP_CTCF_0.0001_32_contrastive_30004_Adam_0_aug_.ckpt" \
#  --output_dir "distances/DP_SP/" \
#  --model_name "SLeNet" \
#  --reference_genome "mm9" \
#  --distance_measure "pairwise" \
#  --bin_size 10000 \
#  --patch_len 224 \
#  --position_stride 10 

#python distance_calculation_all_stats_original_version_middle.py \
#  --mlhic_dataset_path "D4_CTCF_224.json" \
#  --model_ckpt_path "model_outputs/ctcf_final/SLeNet_CTCF_final_0.0001_32_contrastive_30004_Adam_0_aug_.ckpt" \
#  --output_dir "distances/" \
#  --model_name "SLeNet" \
#  --reference_genome "mm10" \
#  --distance_measure "pairwise" \
#  --bin_size 10000 \
  --patch_len 224 
  
python distance_calculation_all_stats_original_version_middle.py \
  --mlhic_dataset_path "Paxip_224.json" \
  --model_ckpt_dir "model_outputs/paxip_final/maxvit_multi/contrastive/" \
  --output_dir "distances/Paxip/true/" \
  --model_name "SMaxViTMulti" \
  --reference_genome "mm10" \
  --bin_size 10000 \
  --patch_len 224 

#python distance_calculation_all_stats_original_version_middle.py \
#  --mlhic_dataset_path "Paxip_224.json" \
#  --model_ckpt_dir "model_outputs/paxip_final/maxvit/supcon/" \
#  --output_dir "distances/Paxip/true/" \
#  --model_name "SMaxVit" \
#  --reference_genome "mm10" \
#  --distance_measure "cosine" \
#  --bin_size 10000 \
#  --patch_len 224 

#python distance_calculation_all_stats.py \
#  --mlhic_dataset_path "D4_CTCF_224.json" \
#  --model_ckpt_dir "model_outputs/ctcf_final/maxvit/supcon/" \
#  --output_dir "distances/CTCF_degron/" \
#  --model_name "SMaxVit" \
#  --reference_genome "mm10" \
#  --distance_measure "cosine" \
#  --bin_size 10000 \
#  --patch_len 224 \
#  --position_stride 10 

#python distance_calculation.py \
#  --mlhic_dataset_path "Paxip_224.json" \
#  --model_ckpt_path "model_outputs/paxip/SLeNet_Paxip_final_0.0001_64_contrastive_30004_Adam_0_aug_.ckpt" \
#  --output_dir "distances/Paxip/" \
#  --model_name "SLeNet" \
#  --reference_genome "mm9" \
#  --distance_measure "pairwise" \
#  --bin_size 10000 \
#  --patch_len 224 \
#  --position_stride 10 

#python distance_calculation.py \
#  --mlhic_dataset_path "Paxip_224.json" \
#  --model_ckpt_path "model_outputs/paxip/SLeNet_Paxip_final_0.0001_8_supcon_30004_Adam_2_aug_.ckpt" \
#  --output_dir "distances/Paxip/" \
#  --model_name "SLeNet" \
#  --reference_genome "mm9" \
#  --distance_measure "cosine" \
#  --bin_size 10000 \
#  --patch_len 224 \
#  --position_stride 10 

#python distance_calculation.py \
#  --mlhic_dataset_path "Paxip_224.json" \
#  --model_ckpt_path "model_outputs/paxip/SLeNet_Paxip_final_0.0001_4_supcon_30004_Adam_2_aug_.ckpt" \
#  --output_dir "distances/Paxip/" \
#  --model_name "SLeNet" \
#  --reference_genome "mm9" \
#  --distance_measure "cosine" \
#  --bin_size 10000 \
#  --patch_len 224 \
#  --position_stride 10 

conda deactivate