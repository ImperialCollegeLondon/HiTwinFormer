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

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss triplet --n_aug 0 --experiment_name D4_CTCF_new --patience 3

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss triplet --n_aug 1 --experiment_name D4_CTCF_new --patience 3

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss triplet --n_aug 2 --experiment_name D4_CTCF_new --patience 3

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss contrastive --n_aug 0 --experiment_name D4_CTCF_new  --patience 3

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss contrastive --n_aug 1 --experiment_name D4_CTCF_new  --patience 3

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss contrastive --n_aug 2 --experiment_name D4_CTCF_new  --patience 3

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss supcon --n_aug 0 --experiment_name D4_CTCF_new --patience 3

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss supcon --n_aug 1 --experiment_name D4_CTCF_new --patience 3

python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.1 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss supcon --n_aug 2 --experiment_name D4_CTCF_new --patience 3 --test_version 0

#python main-v3_with_supcon.py SLeNet D4_CTCF_224.json 0.01 control KO --mask True --outpath /home/tt920/HiTwinFormer/model_outputs/ --epoch_enforced_training 20 --batch_size 128 --loss supcon --n_aug 4 --experiment_name D4_CTCF_new --patience 3


conda deactivate
