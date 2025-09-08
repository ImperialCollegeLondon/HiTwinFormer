#!/bin/bash
#SBATCH --job-name=Twins
#SBATCH --partition=gpu 
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=32g
#SBATCH --output=log_%j.log 
#SBATCH --error=error_%j.err 
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=tt920@lms.mrc.ac.uk

module load miniconda3
conda activate hisinet_test

cd /home/tt920/HiTwinFormer

# Pairwise Contrastive

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 5 --batch_size 128 --loss contrastive --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 5 --batch_size 64 --loss contrastive --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 5 --batch_size 32 --loss contrastive --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 5 --batch_size 16 --loss contrastive --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 5 --batch_size 8 --loss contrastive --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 5 --batch_size 4 --loss contrastive --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

# Triplet

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 128 --loss triplet --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 64 --loss triplet --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 32 --loss triplet --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 16 --loss triplet --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 8 --loss triplet --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 4 --loss triplet --n_aug 0 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

# Supcon

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 64 --loss supcon --n_aug 2 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 32 --loss supcon --n_aug 2 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 16 --loss supcon --n_aug 2 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 8 --loss supcon --n_aug 2 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

python main_final.py SLeNet D4_CTCF_224.json 0.0001 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ctcf_final/ --epoch_enforced_training 10 --batch_size 4 --loss supcon --n_aug 2 --experiment_name D4_CTCF_new --optimiser Adam --patience 3 --use_wandb

conda deactivate