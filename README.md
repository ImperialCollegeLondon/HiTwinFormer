# HiTwinFormer

HiTwinFormer is a workflow for learning robust representations of Hi-C contact maps with Siamese architectures (e.g., LeNet, MaxViT) and using those embeddings to quantify structural similarity across the genome.  
It is an **extension project of Twins** (https://www.nature.com/articles/s41467-023-40547-9), designed to test **different loss functions, backbones, and augmentation strategies** for Hi-C representation learning.

## Workflow

1. **Create conda environment**

create conda environment with environment.yml file using: conda env create -f environment.yml

2. **Create mlhic dataset**

create mlhic data with `generate_mlhic.py` script

3. **Train models**

run `main-v3_with_supcon.py` to generate model. I recommend doing pairwise loss with 0 augmentations and varying batch sizes using both LeNet and MaxViT. Note this require wandb for training - I highly recommend using it - will remove the requirement soon.

4. **Extract features**

run `make_feature_maps.py` to make feature maps

5. **Calculate embedding distances**

run `distance_calculation_all_stats_original_version_middle.py` to generate csv's with embedding distances to simplify downstream analysis

5. **Analyse results**

Now, you are ready to look at the results of the model with `Example_analysis.ipynb`


