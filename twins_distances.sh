#!/bin/bash
#SBATCH --job-name=Twins_distances
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128g
#SBATCH --output=log_%j.log 
#SBATCH --error=error_%j.err 
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=tt920@lms.mrc.ac.uk

module load miniconda3
conda activate hisinet_test

cd /home/tt920/HiTwinFormer

# Run distance calculation with dynamic output CSV path
python distance_calculation.py \
  --mlhic_dataset_path "D4_CTCF_224.json" \
  --model_ckpt_path "model_outputs/SMaxVit_D4_CTCF_new_0.01_32_contrastive_30004_0_aug_.ckpt" \
  --output_dir "distances/" \
  --model_name "SMaxVit" \
  --reference_genome "mm10" \
  --distance_measure "pairwise" \
  --bin_size 10000 \
  --patch_len 224 \
  --position_stride 10 

python distance_calculation.py \
  --mlhic_dataset_path "D4_CTCF_224.json" \
  --model_ckpt_path "model_outputs/SMaxVit_D4_CTCF_new_0.01_16_contrastive_30004_0_aug_.ckpt" \
  --output_dir "distances/" \
  --model_name "SMaxVit" \
  --reference_genome "mm10" \
  --distance_measure "pairwise" \
  --bin_size 10000 \
  --patch_len 224 \
  --position_stride 10 
  
python distance_calculation.py \
  --mlhic_dataset_path "D4_CTCF_224.json" \
  --model_ckpt_path "model_outputs/SMaxVit_D4_CTCF_new_0.01_16_contrastive_30004_1_aug_.ckpt" \
  --output_dir "distances/" \
  --model_name "SMaxVit" \
  --reference_genome "mm10" \
  --distance_measure "pairwise" \
  --bin_size 10000 \
  --patch_len 224 \
  --position_stride 10 
  
python distance_calculation.py \
  --mlhic_dataset_path "D4_CTCF_224.json" \
  --model_ckpt_path "model_outputs/SMaxVit_D4_CTCF_new_0.01_32_triplet_30004_0_aug_.ckpt" \
  --output_dir "distances/" \
  --model_name "SMaxVit" \
  --reference_genome "mm10" \
  --distance_measure "pairwise" \
  --bin_size 10000 \
  --patch_len 224 \
  --position_stride 10 

python distance_calculation.py \
  --mlhic_dataset_path "D4_CTCF_224.json" \
  --model_ckpt_path "model_outputs/SMaxVit_D4_CTCF_new_0.01_16_triplet_30004_1_aug_.ckpt" \
  --output_dir "distances/" \
  --model_name "SMaxVit" \
  --reference_genome "mm10" \
  --distance_measure "pairwise" \
  --bin_size 10000 \
  --patch_len 224 \
  --position_stride 10 

