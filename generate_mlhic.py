import os
import json
from HiSiNet.HiCDatasetClass import HiCDatasetDec, reference_genomes

# Get list of chromosomes from the reference genome
reference_genome_index = "mm10" # Options are hg19, hg38, mm9, mm10, FlyBasev6.07
_, chromsizes = reference_genomes[reference_genome_index]
all_chroms = list(chromsizes.keys())  # ['chr1','chr2','chr3',…, 'chrX','chrY','1']

# Define chrom splits
default_exclude = ['chrY','chrX', 'Y', 'X', 'chrM', 'M']
test_chroms = ['2', 'chr2']
val_chroms = ['18', 'chr18']
resolution = 10000
tile_size = 224

exclude_for_train = default_exclude + [c for c in test_chroms + val_chroms]
exclude_for_test = default_exclude + [c for c in all_chroms if c not in test_chroms]
exclude_for_val = default_exclude + [c for c in all_chroms if c not in val_chroms]

# Define your data (remember they all need to have a consistent order)
replicate_ids = ["R1", "R2", "R3", "R4"] # define which replicate
condition_ids = [0, 0, 1, 1]  # KO=1, control=0
input_data_files = [
    "/home/tt920/mnt/network/lymphdev/Tim/GSM4386026_hic_d4_ctl_rep1.hic", 
    "/home/tt920/mnt/network/lymphdev/Tim/GSM4386027_hic_d4_ctl_rep2.hic",
    "/home/tt920/mnt/network/lymphdev/Tim/GSM4386028_hic_d4_aux_rep1.hic",
    "/home/tt920/mnt/network/lymphdev/Tim/GSM4386029_hic_d4_aux_rep2.hic"
]
output_data_folder = "/home/tt920/mnt/scratch/tt920/mlhic/"
json_file_name = f"D4_CTCF_{tile_size}.json" # where the input paths for the ML model are saved

# Prepare output structure
json_dict = {
    "control": {
        "reference": reference_genome_index,
        "training": [],
        "validation": [],
        "test": []
    },
    "KO": {
        "reference": reference_genome_index,
        "training": [],
        "validation": [],
        "test": []
    }
}

for i in range(len(input_data_files)):
    input_file = input_data_files[i]
    replicate = replicate_ids[i]
    class_id = condition_ids[i]
    
    # Construct base name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Prepare base arguments
    base_args = ([input_file, replicate, 'KR', 'BP', class_id], resolution, tile_size * resolution)

    all_out = os.path.join(output_data_folder, f"all_{base_name}_chr.mlhic")
    all_ds = HiCDatasetDec(*base_args, exclude_chroms=default_exclude)
    all_ds.save(all_out)
    # Save train
    train_out = os.path.join(output_data_folder, f"train_{base_name}.mlhic")
    train = HiCDatasetDec(*base_args, exclude_chroms=exclude_for_train)
    train.save(train_out)

    # Save test
    test_out = os.path.join(output_data_folder, f"test_{base_name}_chr{test_chroms[0]}.mlhic")
    test = HiCDatasetDec(*base_args, exclude_chroms=exclude_for_test)
    test.save(test_out)

    # Save validation
    val_out = os.path.join(output_data_folder, f"val_{base_name}_chr{val_chroms[0]}.mlhic")
    val = HiCDatasetDec(*base_args, exclude_chroms=exclude_for_val)
    val.save(val_out)

    # Add to json dict
    key = "KO" if class_id == 1 else "control" # This is where we use
    json_dict[key]["training"].append(train_out)
    json_dict[key]["validation"].append(val_out)
    json_dict[key]["test"].append(test_out)

# Save json
with open(json_file_name, "w") as f:
    json.dump(json_dict, f, indent=2)

