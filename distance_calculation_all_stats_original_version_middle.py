import argparse
import torch
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, Subset
import json
from scipy import stats
from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset
from HiSiNet.reference_dictionaries import reference_genomes
from HiSiNet import models

def summarize_distances(df: pd.DataFrame, chr_col: str = "Chromosome", start_col: str = "Start", label_col: str = "labels_siam") -> pd.DataFrame:
    grouped = (
        df
        .groupby([chr_col, start_col, label_col])['distances']
        .agg(['mean', 'count', 'std'])
        .reset_index()
    )
    
    pivot = grouped.pivot_table(
        index=[chr_col, start_col],
        columns=label_col,
        values=['mean', 'count', 'std']
    )
    
    pivot.columns = [f"{stat}_{lab}" for stat, lab in pivot.columns]
    pivot = pivot.reset_index()
    
    pivot['distance_simple'] = (pivot['mean_between'] - pivot['mean_within']).clip(lower=0)
    
    z = stats.norm.ppf(0.975)
    for grp in ['within','between']:
        mu = pivot[f'mean_{grp}']
        sd = pivot[f'std_{grp}']
        n = pivot[f'count_{grp}']
        se = sd / np.sqrt(n)
        pivot[f'ci_low_{grp}'] = mu - z * se
        pivot[f'ci_high_{grp}'] = mu + z * se
    
    low_b, high_b = pivot['ci_low_between'], pivot['ci_high_between']
    low_w, high_w = pivot['ci_low_within'], pivot['ci_high_within']
    overlap = (low_b <= high_w) & (high_b >= low_w)
    diff_nonoverlap = (low_b - high_w).clip(lower=0)
    pivot['distance_ci'] = np.where(overlap, 0.0, diff_nonoverlap)
    
    return pivot

def process_checkpoint(args, ckpt_path):
    ckpt_basename = os.path.splitext(os.path.basename(ckpt_path))[0]
    output_raw_csv = os.path.join(args.output_dir, f"{ckpt_basename}_{args.distance_measure}_distances_raw.csv")
    output_summary_csv = os.path.join(args.output_dir, f"{ckpt_basename}_{args.distance_measure}_distances_summary.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_class = getattr(models, args.model_name)
    model = model_class(mask=True, image_size=args.patch_len)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    
    with open(args.mlhic_dataset_path) as json_file:
        dataset = json.load(json_file)
    
    all_paths = []
    for key in dataset:
        all_paths.extend(dataset[key].get("all", []))
    
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
    
    def test_model(model, dataloader):
        """Modified to return distances only for tile start coordinates"""
        all_distances = []
        all_labels = []
        
        print("Running model inference...")
        for batch_idx, (x1, x2, y) in enumerate(dataloader):
            if batch_idx % 5 == 0:
                print(f" Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            with torch.no_grad():
                o1, o2 = model(x1, x2)
                
                if args.distance_measure.lower() == "cosine":
                    dists = 1 - F.cosine_similarity(o1, o2)
                elif args.distance_measure.lower() == "pairwise":
                    dists = F.pairwise_distance(o1, o2)
                else:
                    raise ValueError(f"Unknown distance measure: {args.distance_measure}")
                
                # KEY CHANGE: No expansion - keep one distance per tile
                all_distances.append(dists.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        return np.concatenate(all_distances), np.concatenate(all_labels)
    
    all_data = []
    
    for chrom in all_chroms:
        print(f"\nProcessing chromosome {chrom}...")
        siamese_subset = subset_by_chromosome(siamese, chrom)
        
        if len(siamese_subset) == 0:
            print(f"No data found for chromosome {chrom}, skipping...")
            continue
        
        dataloader = DataLoader(siamese_subset, batch_size=args.batch_size, 
                               sampler=SequentialSampler(siamese_subset))
        
        distances, labels_siam = test_model(model, dataloader)
        
        subset_indices = siamese_subset.indices
        pd_dict = siamese.get_genomic_positions(append="chr")
        subset_pd_dict = {key: [val[i] for i in subset_indices] for key, val in pd_dict.items()}
        
        start_positions = subset_pd_dict["Start"]
        chromosomes = subset_pd_dict["Chromosome"]
        subset_labels = [siamese.labels[i] for i in subset_indices]
        
        # Assign distance to the middle bin of the patch
        middle_positions = [start + (args.bin_size * args.patch_len // 2) for start in start_positions]
        
        assert len(middle_positions) == len(distances), f"Mismatch: {len(middle_positions)} positions vs {len(distances)} distances"
        
        df = pd.DataFrame({
            "Chromosome": chromosomes,
            "Start": middle_positions,  # middle bin
            "distances": distances,
            "labels": subset_labels,
            "labels_siam": np.where(labels_siam==0, "within", "between")
        })
        
        # Apply position stride if specified (subsample tile start positions)
        if args.position_stride > 1:
            unique_positions = sorted(df["Start"].unique())
            selected_positions = unique_positions[::args.position_stride]
            df = df[df["Start"].isin(selected_positions)].reset_index(drop=True)
        
        all_data.append(df)
    
    if not all_data:
        print(f"Skipping {ckpt_path} — no data found")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['model_name'] = args.model_name
    combined_df['distance_measure'] = args.distance_measure
    combined_df['bin_size'] = args.bin_size
    combined_df['patch_len'] = args.patch_len
    combined_df['position_stride'] = args.position_stride
    
    combined_df.to_csv(output_raw_csv, index=False)
    print(f"Saved raw distances to: {output_raw_csv}")
    
    summary_df = summarize_distances(combined_df)
    summary_df['model_name'] = args.model_name
    summary_df['distance_measure'] = args.distance_measure
    summary_df['bin_size'] = args.bin_size
    summary_df['patch_len'] = args.patch_len
    summary_df['position_stride'] = args.position_stride
    
    summary_df.to_csv(output_summary_csv, index=False)
    print(f"Saved summary distances to: {output_summary_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlhic_dataset_path", type=str, required=True)
    parser.add_argument("--model_ckpt_path", type=str, help="Path to single checkpoint")
    parser.add_argument("--model_ckpt_dir", type=str, help="Path to folder containing checkpoints")
    parser.add_argument("--reference_genome", type=str, default="mm9")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="SLeNet")
    parser.add_argument("--distance_measure", type=str, choices=["pairwise", "cosine"], default="pairwise")
    parser.add_argument("--bin_size", type=int, default=10000)
    parser.add_argument("--patch_len", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--position_stride", type=int, default=1)
    
    args = parser.parse_args()
    
    if not args.model_ckpt_path and not args.model_ckpt_dir:
        parser.error("Must provide either --model_ckpt_path or --model_ckpt_dir")
    
    if args.model_ckpt_dir:
        ckpt_files = [os.path.join(args.model_ckpt_dir, f) 
                     for f in os.listdir(args.model_ckpt_dir) 
                     if f.endswith((".ckpt", ".pth"))]
        ckpt_files.sort()
        
        for ckpt in ckpt_files:
            print(f"\n=== Processing checkpoint: {ckpt} ===")
            process_checkpoint(args, ckpt)
    else:
        process_checkpoint(args, args.model_ckpt_path)

if __name__ == "__main__":
    main()