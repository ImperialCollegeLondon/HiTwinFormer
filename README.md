# HiTwinFormer

workflow:

create conda environment with environment.yml file using: conda env create -f environment.yml

you might have to make a feature_maps and model_ouputs folder, not sure if they are created automatically in the scripts

create mlhic data with `generate_mlhic.py script`

run `main-v3_with_supcon.py` to generate model. I recommend doing triplet loss with 0 augmentations for now. Note this require wandb, will make non wandb script later.

run `make_feature_maps.py` to make feature maps

Now, you are ready to look at the results of the model with `visualise_changes.ipynb`


