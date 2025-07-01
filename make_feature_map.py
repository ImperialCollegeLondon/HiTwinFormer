import argparse
from HiSiNet.HiCDatasetClass import HiCDatasetDec, PairOfDatasets
from HiSiNet.reference_dictionaries import reference_genomes
from HiSiNet import models
import torch

# 1) Change your argparse to two groups:
parser = argparse.ArgumentParser(description='Siamese network feature mapping')
parser.add_argument('--model_name',    type=str,   required=True)
parser.add_argument('--model_file',    type=str,   required=True)
parser.add_argument('--save_path',     type=str,   required=True)
parser.add_argument('--reference_genome', type=str, required=True)
parser.add_argument(
    '--mlhic_paths', nargs='+', help='paths to .mlhic files', required=True
)

args = parser.parse_args()

# 2) Load your model (as before)…
model = eval(f"models.{args.model_name}")(mask=True, image_size=224)
checkpoint = torch.load(args.model_file, map_location='cpu')
if 'mask' in checkpoint and checkpoint['mask'].ndim == 3:
    checkpoint['mask'] = checkpoint['mask'].squeeze(0)
model.load_state_dict(checkpoint)
model.eval()

# 3) Load the HiCDatasetDec objects
list_of_HiCDatasets = [HiCDatasetDec.load(p) for p in args.mlhic_paths]

# 4) Pass the list into PairOfDatasets:
paired = PairOfDatasets(
    list_of_HiCDatasets,
    model,
    reference=reference_genomes[args.reference_genome]
)

# 5) (Optional) If you only care about the maps, you can drop the raw data
del paired.data
del paired.labels

paired.save(args.save_path)
