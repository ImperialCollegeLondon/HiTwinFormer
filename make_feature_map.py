import argparse
from HiSiNet.HiCDatasetClass import HiCDatasetDec, PairOfDatasetsModified
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
    '--control_paths', nargs='+', help='paths to control .mlhic files', required=True
)
parser.add_argument(
    '--ko_paths', nargs='+', help='paths to KO .mlhic files', required=True
)
args = parser.parse_args()

# 2) Load your model (as before)…
model = eval(f"models.{args.model_name}")(mask=True, image_size=224)
checkpoint = torch.load(args.model_file, map_location='cpu')
if 'mask' in checkpoint and checkpoint['mask'].ndim == 3:
    checkpoint['mask'] = checkpoint['mask'].squeeze(0)
model.load_state_dict(checkpoint)
model.eval()

# 3) Load the HiCDatasetDec objects **in the order you want them subtracted**:
control_ds = [HiCDatasetDec.load(p) for p in args.control_paths]
ko_ds      = [HiCDatasetDec.load(p) for p in args.ko_paths]

# 4) Concatenate them so that index 0…N_control-1 are all WT,
#    and index N_control… end are the KO datasets:
list_of_HiCDatasets = control_ds + ko_ds

# 5) Pass that ordered list into PairOfDatasets:
paired = PairOfDatasetsModified(
    list_of_HiCDatasets,
    model,
    reference=reference_genomes[args.reference_genome]
)

# 6) (Optional) If you only care about the maps, you can drop the raw data
del paired.data
del paired.labels

paired.save(args.save_path)
