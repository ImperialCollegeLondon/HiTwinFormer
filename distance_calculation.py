import argparse
import torch
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, Subset
import json
from HiSiNet.HiCDatasetClass import HiCDataset, HiCDatasetDec, SiameseHiCDataset, PairOfDatasets
from HiSiNet.reference_dictionaries import reference_genomes
from HiSiNet import models


def main():
    parser = argparse.ArgumentParser(description="Calculate chromosome-wide embedding distances from a Siamese Hi-C model")
    
    parser.add_argument("--mlhic_dataset_path", type=str, required=True,
                        help="Path to json file")
    parser.add_argument("--model_ckpt_path", type=str, required=True,
                        help="Path to trained Siamese model checkpoint")
    parser.add_argument("--reference_genome", type=str, default="mm9",
                        help="Reference genome key, e.g. 'mm9'")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="folder to save the output CSV file")
    parser.add_argument("--model_name", type=str, default="SLeNet",
                        help="Name of model in HiSiNet.models")
    parser.add_argument("--distance_measure", type=str, choices=["pairwise", "cosine"], default="pairwise",
                        help="Distance measure: pairwise or cosine")
    parser.add_argument("--bin_size", type=int, default=10000,
                        help="Hi-C resolution in bp")
    parser.add_argument("--patch_len", type=int, default=224,
                        help="Model patch length in bins")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for model inference")
    parser.add_argument("--position_stride", type=int, default=10,
                        help="Subsample every Nth position (1 = no subsampling)")

    args = parser.parse_args()
    ckpt_basename = os.path.splitext(os.path.basename(args.model_ckpt_path))[0]
    output_csv_path = os.path.join(args.output_dir, f"{ckpt_basename}_{args.distance_measure}_distances.csv")
    
    # Load JSON once
    
    
    model_class = getattr(models, args.model_name)
    model = model_class(mask=True)
    model.load_state_dict(torch.load(args.model_ckpt_path, map_location=torch.device("cpu")))
    model.eval()

    with open(args.mlhic_dataset_path) as json_file:
        dataset = json.load(json_file)
    all_paths = []
    for key in dataset:
        all_paths.extend(dataset[key].get("all", []))

    # Load all datasets
    mlhic_dataset = [HiCDatasetDec.load(p) for p in all_paths]
    siamese = SiameseHiCDataset(mlhic_dataset, reference=reference_genomes[args.reference_genome])
    
    def normalize_chrom(chrom):
        return chrom[3:] if chrom.startswith("chr") else chrom
    
    def subset_by_chromosome(dataset, target_chrom):
        target_chrom_norm = normalize_chrom(target_chrom)
        indices = [
            i for i, (_, chrom) in enumerate(dataset.pos)
            if normalize_chrom(chrom) == target_chrom_norm
        ]
        print(f"Found {len(indices)} samples for chromosome {target_chrom}")
        return Subset(dataset, indices)
    
    all_chroms = sorted(list(set([chrom for _, chrom in siamese.pos])))
    print(f"Found chromosomes: {all_chroms}")
    
    def test_model(model, dataloader, patch_len=args.patch_len):
        all_distances = []
        all_labels = []
        print("Running model inference...")
        
        for batch_idx, (x1, x2, y) in enumerate(dataloader):
            if batch_idx % 5 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            y = y.float()
            with torch.no_grad():
                o1, o2 = model(x1, x2)
                if args.distance_measure.lower() == "cosine":
                    dists = 1 - F.cosine_similarity(o1, o2)
                elif args.distance_measure.lower() == "pairwise":
                    dists = F.pairwise_distance(o1, o2)
                else:
                    raise ValueError(f"Unknown distance measure: {args.distance_measure}")
            
            expanded_dists = np.repeat(dists.cpu().numpy(), patch_len)
            expanded_labels = np.repeat(y.cpu().numpy(), patch_len)
            all_distances.append(expanded_dists)
            all_labels.append(expanded_labels)
        
        return np.concatenate(all_distances), np.concatenate(all_labels)
    
    all_data = []
    
    for chrom in all_chroms:
        print(f"\nProcessing chromosome {chrom}...")
        siamese_subset = subset_by_chromosome(siamese, chrom)
        
        if len(siamese_subset) == 0:
            print(f"No data found for chromosome {chrom}, skipping...")
            continue
        
        dataloader = DataLoader(siamese_subset, batch_size=args.batch_size, sampler=SequentialSampler(siamese_subset))
        
        distances, labels_siam = test_model(model, dataloader, patch_len=args.patch_len)
        
        print("Processing genomic positions...")
        
        subset_indices = siamese_subset.indices
        pd_dict = siamese.get_genomic_positions(append="chr")
        
        subset_pd_dict = {
            key: [val[i] for i in subset_indices]
            for key, val in pd_dict.items()
        }
        
        start_positions = subset_pd_dict["Start"]
        chromosomes = subset_pd_dict["Chromosome"]
        subset_labels = [siamese.labels[i] for i in subset_indices]
        
        assert len(start_positions) == len(distances) // args.patch_len, "Mismatch in number of patches"
        
        expanded_starts = []
        expanded_chroms = []
        expanded_true_labels = []
        for chrom_name, start, label in zip(chromosomes, start_positions, subset_labels):
            for i in range(args.patch_len):
                expanded_starts.append(start + i * args.bin_size)
                expanded_chroms.append(chrom_name)
                expanded_true_labels.append(label)
        
        df = pd.DataFrame({
            "Chromosome": expanded_chroms,
            "Start": expanded_starts,
            "distances": distances,
            "labels": expanded_true_labels,
            "labels_siam": np.where(labels_siam==0, "within", "between")
        })
        
        if args.position_stride > 1:
            print(f"Subsampling every {args.position_stride} positions for chromosome {chrom}...")
            unique_positions = sorted(df["Start"].unique())
            selected_positions = unique_positions[::args.position_stride]
            df = df[df["Start"].isin(selected_positions)].reset_index(drop=True)
            print(f"Subsampled to {len(df)} rows for chromosome {chrom}")
        
        print(f"Created DataFrame with {len(df)} rows for {chrom}")
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No data found for any chromosome")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined data: {len(combined_df)} total rows across {len(all_data)} chromosomes")
    
    combined_df['model_name'] = args.model_name
    combined_df['distance_measure'] = args.distance_measure
    combined_df['bin_size'] = args.bin_size
    combined_df['patch_len'] = args.patch_len
    combined_df['position_stride'] = args.position_stride
    
    combined_df.to_csv(output_csv_path, index=False)
    print(f"Saved distances to: {output_csv_path}")

    

if __name__ == "__main__":
    main()
