#!/bin/bash
#SBATCH --job-name=Twins_feature_map
#SBATCH --partition=cpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH --output=log_%j.log
#SBATCH --error=error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tt920@lms.mrc.ac.uk

module load miniconda3
conda activate hisinet_test

cd /home/tt920/HiTwinFormer

python make_feature_map.py \
    --model_name SLeNet \
    --model_file /home/tt920/HiTwinFormer/model_outputs/SLeNet_D4_CTCF_new_0.01_128_supcon_30004_4_aug.ckpt \
    --save_path /home/tt920/HiTwinFormer/feature_maps/ctcf_npc_features_supcon_lenet_4_aug.fmlhic \
    --reference_genome mm10 \
    --control_paths \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386026_hic_d4_ctl_rep1_chr.mlhic \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386027_hic_d4_ctl_rep2_chr.mlhic \
    --ko_paths \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386028_hic_d4_aux_rep1_chr.mlhic \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386029_hic_d4_aux_rep2_chr.mlhic

python make_feature_map.py \
    --model_name SLeNet \
    --model_file /home/tt920/HiTwinFormer/model_outputs/SLeNet_D4_CTCF_new_0.01_128_contrastive_30004_0_aug.ckpt \
    --save_path /home/tt920/HiTwinFormer/feature_maps/ctcf_npc_features_contrastive_0_aug.fmlhic \
    --reference_genome mm10 \
    --control_paths \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386026_hic_d4_ctl_rep1_chr.mlhic \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386027_hic_d4_ctl_rep2_chr.mlhic \
    --ko_paths \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386028_hic_d4_aux_rep1_chr.mlhic \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386029_hic_d4_aux_rep2_chr.mlhic

python make_feature_map.py \
    --model_name SLeNet \
    --model_file /home/tt920/HiTwinFormer/model_outputs/SLeNet_D4_CTCF_new_0.01_128_triplet_30004_0_aug.ckpt \
    --save_path /home/tt920/HiTwinFormer/feature_maps/ctcf_npc_features_triplet_0_aug.fmlhic \
    --reference_genome mm10 \
    --control_paths \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386026_hic_d4_ctl_rep1_chr.mlhic \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386027_hic_d4_ctl_rep2_chr.mlhic \
    --ko_paths \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386028_hic_d4_aux_rep1_chr.mlhic \
        /home/tt920/mnt/scratch/tt920/mlhic/all_GSM4386029_hic_d4_aux_rep2_chr.mlhic

#--control_paths \
#        /home/tt920/mnt/scratch/tt920/mlhic/all_CD69negDPWTR3_chr.mlhic \
#        /home/tt920/mnt/scratch/tt920/mlhic/all_CD69negDPWTR4_chr.mlhic \
#    --ko_paths \
#        /home/tt920/mnt/scratch/tt920/mlhic/all_CD69negDPPaxip1KOR1_chr.mlhic \
#        /home/tt920/mnt/scratch/tt920/mlhic/all_CD69negDPPaxip1KOR2_chr.mlhic

conda deactivate