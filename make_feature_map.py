import argparse
from HiSiNet.HiCDatasetClass import HiCDatasetDec, PairOfDatasets
from HiSiNet.reference_dictionaries import reference_genomes
from HiSiNet import models
import torch
import json


# 1) Change your argparse to two groups:
parser = argparse.ArgumentParser(description='Siamese network feature mapping')
parser.add_argument('--model_name',    type=str,   required=True)
parser.add_argument('--model_file',    type=str,   required=True)
parser.add_argument('--save_path',     type=str,   required=True)
parser.add_argument('--reference_genome', type=str, required=True)
parser.add_argument('--json_file', type=str, help='paths to .mlhic files', required=True)
parser.add_argument('--saliency_mapping', action='store_true')
parser.add_argument('--distance_measure', type=str, default= "pairwise")

args = parser.parse_args()

print("started model loading")

# 2) Load your model (as before)…
model = eval(f"models.{args.model_name}")(mask=True, image_size=224)
checkpoint = torch.load(args.model_file, map_location='cpu')
if 'mask' in checkpoint and checkpoint['mask'].ndim == 3:
    checkpoint['mask'] = checkpoint['mask'].squeeze(0)
model.load_state_dict(checkpoint)
model.eval()


print("model loaded")

# 3) Load the HiCDatasetDec objects

json_file = args.json_file

with open(json_file) as json_file:
    dataset = json.load(json_file)
all_paths = []
for key in dataset:
    all_paths.extend(dataset[key].get("all", []))
    
list_of_HiCDatasets = [HiCDatasetDec.load(p) for p in all_paths]

print("dataset loaded and started feature map creation")


# 4) Pass the list into PairOfDatasets:
paired = PairOfDatasets(
    list_of_HiCDatasets,
    model,
    reference=reference_genomes[args.reference_genome],
    compute_saliency=args.saliency_mapping
    #distance_measure=args.distance_measure
)

# 5) (Optional) If you only care about the maps, you can drop the raw data
del paired.data
del paired.labels
del paired.saliency

paired.save(args.save_path)
