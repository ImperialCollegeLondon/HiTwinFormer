import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import rotate
from skimage.transform import resize
import hicstraw
from torch.utils.data import DataLoader, SequentialSampler, Subset
from matplotlib.ticker import MultipleLocator
import pickle
import HiSiNet.models as models
from HiSiNet.HiCDatasetClass import HiCDataset, HiCDatasetDec, SiameseHiCDataset, PairOfDatasets
from HiSiNet.reference_dictionaries import reference_genomes

def calculate_chromosome_distances(
    mlhic_dataset,
    model_ckpt_path,
    reference_genome,
    output_dir,
    model_name="SLeNet",
    distance_measure="pairwise",
    bin_size=10000,
    patch_len=224,
    batch_size=100,
    position_stride=1,
):
    """
    Calculate chromosome-wide embedding distances from a Siamese Hi-C model and save to CSV.
    
    Parameters:
    -----------
    mlhic_dataset : list
        List of loaded mlhic
    model_ckpt_path : str
        Path to trained Siamese model checkpoint
    reference_genome : dict
        Reference genome dictionary (e.g., reference_genomes["mm9"])
    output_dir : str
        Path to output directory
    model_name : str
        Name of model in HiSiNet.models
    distance_measure : str, default "pairwise"
        Distance measure: "pairwise" or "cosine"
    bin_size : int, default 10000
        Hi-C resolution in bp
    patch_len : int, default 224
        Model patch length in bins
    batch_size : int, default 100
        Batch size for model inference
    position_stride : int, default 1
        Subsample every Nth position (1 = no subsampling)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Chromosome, Start, distances, labels, labels_siam
    """
    
    ckpt_basename = os.path.splitext(os.path.basename(model_ckpt_path))[0]
    output_csv_path = os.path.join(output_dir, f"{ckpt_basename}_{distance_measure}_distances.csv")

    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
    print(f"Loading model from: {model_ckpt_path}")
    
    # --- Load model ---
    model_class = getattr(models, model_name)
    model = model_class(mask=True)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=torch.device("cpu")))
    model.eval()
    
    # --- Load data ---
    siamese = SiameseHiCDataset(mlhic_dataset, reference=reference_genomes[reference_genome])
    
    # Add chromosome subsetting functionality
    def normalize_chrom(chrom):
        """Strip 'chr' if present to normalize chromosome names."""
        return chrom[3:] if chrom.startswith("chr") else chrom
    
    def subset_by_chromosome(dataset, target_chrom):
        # Normalize the target too
        target_chrom_norm = normalize_chrom(target_chrom)
        indices = [
            i for i, (_, chrom) in enumerate(dataset.pos)
            if normalize_chrom(chrom) == target_chrom_norm
        ]
        print(f"Found {len(indices)} samples for chromosome {target_chrom}")
        return Subset(dataset, indices)
    
    # Get all unique chromosomes
    all_chroms = sorted(list(set([chrom for _, chrom in siamese.pos])))
    print(f"Found chromosomes: {all_chroms}")
    
    # --- Test model function ---
    def test_model(model, dataloader, patch_len=224):
        """Test Siamese model and return distances"""
        all_distances = []
        all_labels = []
        print("Running model inference...")
        
        for batch_idx, (x1, x2, y) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            y = y.float()
            with torch.no_grad():
                o1, o2 = model(x1, x2)
                if distance_measure.lower() == "cosine":
                    dists = 1 - F.cosine_similarity(o1, o2)
                elif distance_measure.lower() == "pairwise":
                    dists = F.pairwise_distance(o1, o2)
                else:
                    raise ValueError(f"Unknown distance measure: {distance_measure}")
            
            # Expand each distance/label for all bins in the patch
            expanded_dists = np.repeat(dists.cpu().numpy(), patch_len)
            expanded_labels = np.repeat(y.cpu().numpy(), patch_len)
            all_distances.append(expanded_dists)
            all_labels.append(expanded_labels)
        
        return np.concatenate(all_distances), np.concatenate(all_labels)
    
    # Process each chromosome and collect all data
    all_data = []
    
    for chrom in all_chroms:
        print(f"\nProcessing chromosome {chrom}...")
        
        # Subset the dataset by chromosome
        from torch.utils.data import Subset
        siamese_subset = subset_by_chromosome(siamese, chrom)
        
        if len(siamese_subset) == 0:
            print(f"No data found for chromosome {chrom}, skipping...")
            continue
            
        dataloader = DataLoader(siamese_subset, batch_size=batch_size, sampler=SequentialSampler(siamese_subset))
        
        # --- Run model testing ---
        distances, labels_siam = test_model(model, dataloader, patch_len=patch_len)
        
        print("Processing genomic positions...")
        
        # --- Get genomic positions and expand them ---
        subset_indices = siamese_subset.indices
        
        # Use original dataset method to get all positions
        pd_dict = siamese.get_genomic_positions(append="chr")
        
        # Filter positions using subset indices
        subset_pd_dict = {
            key: [val[i] for i in subset_indices]
            for key, val in pd_dict.items()
        }
        
        start_positions = subset_pd_dict["Start"]
        chromosomes = subset_pd_dict["Chromosome"]
        
        # Get labels for the subset
        subset_labels = [siamese.labels[i] for i in subset_indices]
        
        # Sanity check
        assert len(start_positions) == len(distances) // patch_len, "Mismatch in number of patches"
        
        # Expand genomic positions
        expanded_starts = []
        expanded_chroms = []
        expanded_true_labels = []
        for chrom_name, start, label in zip(chromosomes, start_positions, subset_labels):
            for i in range(patch_len):
                expanded_starts.append(start + i * bin_size)
                expanded_chroms.append(chrom_name)
                expanded_true_labels.append(label)
        
        # --- Construct DataFrame ---
        df = pd.DataFrame({
            "Chromosome": expanded_chroms,
            "Start": expanded_starts,
            "distances": distances,
            "labels": expanded_true_labels,
            "labels_siam": np.where(labels_siam==0, "within", "between")
        })
        
        # Apply position stride subsampling if specified
        if position_stride > 1:
            print(f"Subsampling every {position_stride} positions for chromosome {chrom}...")
            unique_positions = sorted(df["Start"].unique())
            selected_positions = unique_positions[::position_stride]
            df = df[df["Start"].isin(selected_positions)].reset_index(drop=True)
            print(f"Subsampled to {len(df)} rows for chromosome {chrom}")
        
        print(f"Created DataFrame with {len(df)} rows for {chrom}")
        all_data.append(df)
    
    # Combine all chromosome data
    if not all_data:
        raise ValueError("No data found for any chromosome")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined data: {len(combined_df)} total rows across {len(all_data)} chromosomes")
    
    # Add metadata columns
    combined_df['model_name'] = model_name
    combined_df['distance_measure'] = distance_measure
    combined_df['bin_size'] = bin_size
    combined_df['patch_len'] = patch_len
    combined_df['position_stride'] = position_stride
    
    # Save to CSV
    combined_df.to_csv(output_csv_path, index=False)
    print(f"Saved distances to: {output_csv_path}")
    
    #return combined_df

def plot_chromosome_distances(
    mlhic_dataset=None,
    model_ckpt_path=None,
    reference_genome=None,
    model_name="SLeNet",
    target_chrom=None,
    loss_function="contrastive",
    distance_measure="pairwise",
    mb_start=None,
    mb_end=None,
    bin_size=10000,
    patch_len=224,
    position_stride=10,
    batch_size=100,
    figsize=(30, 5),
    title=None,
    major_tick_interval=5,
    font_scale=1.5,
    # New parameters for CSV input
    csv_path=None,
    csv_data=None
):
    """
    Plot chromosome-wide embedding distances from a Siamese Hi-C model.
    Can either calculate distances on-the-fly or use pre-calculated data from CSV.
    
    Parameters:
    -----------
    mlhic_dataset : list, optional
        List of loaded mlhic. Required if csv_path and csv_data are None.
    model_ckpt_path : str, optional
        Path to trained Siamese model checkpoint. Required if csv_path and csv_data are None.
    reference_genome : dict, optional
        Reference genome dictionary. Required if csv_path and csv_data are None.
    model_name : str
        Name of model in HiSiNet.models
    target_chrom : str, optional
        Target chromosome to plot. If None, plots all chromosomes in separate subplots.
    loss_function : str
        Loss function for title generation
    distance_measure : str, default "pairwise"
        Distance measure: "pairwise" or "cosine"
    mb_start : float, optional
        Start position in Mb. If None, starts from chromosome beginning.
    mb_end : float, optional
        End position in Mb. If None, goes to chromosome end.
    bin_size : int, default 10000
        Hi-C resolution in bp
    patch_len : int, default 224
        Model patch length in bins
    position_stride : int, default 1
        Subsampling stride for plotting. Want 10 if showing all chrom at once. 
    batch_size : int, default 100
        Batch size for model inference
    figsize : tuple, default (30, 5)
        Figure size (width, height) - for single chromosome or per chromosome when plotting all
    title : str, optional
        Plot title. If None, uses default based on distance measure.
    major_tick_interval : float, default 5
        Major tick interval in Mb for x-axis
    font_scale : float, default 1.5
        Seaborn font scale
    csv_path : str, optional
        Path to CSV file with pre-calculated distances
    csv_data : pd.DataFrame, optional
        Pre-calculated distances DataFrame
    """
    
    # Set seaborn context
    sns.set_context("notebook", font_scale=font_scale)

     # Add chromosome subsetting functionality
    def normalize_chrom(chrom):
        """Strip 'chr' if present to normalize chromosome names."""
        return chrom[3:] if chrom.startswith("chr") else chrom
    
    def subset_by_chromosome(dataset, target_chrom):
        # Normalize the target too
        target_chrom_norm = normalize_chrom(target_chrom)
        indices = [
            i for i, (_, chrom) in enumerate(dataset.pos)
            if normalize_chrom(chrom) == target_chrom_norm
        ]
        print(f"Found {len(indices)} samples for chromosome {target_chrom}")
        return Subset(dataset, indices)
    
    def chrom_sort_key(chrom):
        c = normalize_chrom(str(chrom)).lower()
        if c == "x":
            return 1000
        elif c == "y":
            return 1001
        elif c in ("m", "mt"):
            return 1002
        try:
            return int(c)
        except ValueError:
            return float('inf')
    
    # Determine data source
    if csv_path is not None:
        print(f"Loading pre-calculated distances from: {csv_path}")
        df = pd.read_csv(csv_path)
    elif csv_data is not None:
        print("Using provided pre-calculated distances DataFrame")
        df = csv_data.copy()
    else:
        # Calculate distances on-the-fly (original behavior)
        if mlhic_dataset is None or model_ckpt_path is None or reference_genome is None:
            raise ValueError("Must provide either csv_path/csv_data or mlhic_dataset/model_ckpt_path/reference_genome")
        
        print(f"Calculating distances on-the-fly...")
        print(f"Loading model from: {model_ckpt_path}")
        
        # --- Load model ---
        model_class = getattr(models, model_name)
        model = model_class(mask=True)
        model.load_state_dict(torch.load(model_ckpt_path, map_location=torch.device("cpu")))
        model.eval()
        
        # --- Load data and calculate distances ---
        siamese = SiameseHiCDataset(mlhic_dataset, reference=reference_genomes[reference_genome])
        
        # Get all unique chromosomes if plotting all
        if target_chrom is None:
            all_chroms = sorted(list(set([chrom for _, chrom in siamese.pos])), key=chrom_sort_key)
            print(f"Found chromosomes: {all_chroms}")
            print("Calculating distances for all chromosomes in separate subplots")
        else:
            all_chroms = [target_chrom]
            print(f"Calculating distances for single chromosome: {target_chrom}")

        # --- Test model function ---
        def test_model(model, dataloader, patch_len=224):
            """Test Siamese model and return distances"""
            all_distances = []
            all_labels = []
            print("Running model inference...")
            
            for batch_idx, (x1, x2, y) in enumerate(dataloader):
                if batch_idx % 5 == 0:
                    print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
                
                y = y.float()
                with torch.no_grad():
                    o1, o2 = model(x1, x2)
                    if distance_measure.lower() == "cosine":
                        dists = 1 - F.cosine_similarity(o1, o2)
                    elif distance_measure.lower() == "pairwise":
                        dists = F.pairwise_distance(o1, o2)
                    else:
                        raise ValueError(f"Unknown distance measure: {distance_measure}")
                
                # Expand each distance/label for all bins in the patch
                expanded_dists = np.repeat(dists.cpu().numpy(), patch_len)
                expanded_labels = np.repeat(y.cpu().numpy(), patch_len)
                all_distances.append(expanded_dists)
                all_labels.append(expanded_labels)
            
            return np.concatenate(all_distances), np.concatenate(all_labels)
        
        # Process each chromosome and collect all data
        all_data = []
        
        for chrom in all_chroms:
            print(f"\nProcessing chromosome {chrom}...")
            
            # Subset the dataset by chromosome
            from torch.utils.data import Subset
            siamese_subset = subset_by_chromosome(siamese, chrom)
            
            if len(siamese_subset) == 0:
                print(f"No data found for chromosome {chrom}, skipping...")
                continue
                
            dataloader = DataLoader(siamese_subset, batch_size=batch_size, sampler=SequentialSampler(siamese_subset))
            
            # --- Run model testing ---
            distances, labels_siam = test_model(model, dataloader, patch_len=patch_len)
            
            print("Processing genomic positions...")
            
            # --- Get genomic positions and expand them ---
            subset_indices = siamese_subset.indices
            
            # Use original dataset method to get all positions
            pd_dict = siamese.get_genomic_positions(append="chr")
            
            # Filter positions using subset indices
            subset_pd_dict = {
                key: [val[i] for i in subset_indices]
                for key, val in pd_dict.items()
            }
            
            start_positions = subset_pd_dict["Start"]
            chromosomes = subset_pd_dict["Chromosome"]
            
            # Get labels for the subset
            subset_labels = [siamese.labels[i] for i in subset_indices]
            
            # Sanity check
            assert len(start_positions) == len(distances) // patch_len, "Mismatch in number of patches"
            
            # Expand genomic positions
            expanded_starts = []
            expanded_chroms = []
            expanded_true_labels = []
            for chrom_name, start, label in zip(chromosomes, start_positions, subset_labels):
                for i in range(patch_len):
                    expanded_starts.append(start + i * bin_size)
                    expanded_chroms.append(chrom_name)
                    expanded_true_labels.append(label)
            
            # --- Construct DataFrame ---
            chrom_df = pd.DataFrame({
                "Chromosome": expanded_chroms,
                "Start": expanded_starts,
                "distances": distances,
                "labels": expanded_true_labels,
                "labels_siam": np.where(labels_siam==0, "within", "between")
            })
            
            print(f"Created DataFrame with {len(chrom_df)} rows for {chrom}")
            all_data.append(chrom_df)
        
        # Combine all chromosome data
        if not all_data:
            raise ValueError("No data found for any chromosome")
        
        df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined data: {len(df)} total rows across {len(all_data)} chromosomes")
    
    # Now proceed with plotting using the df DataFrame
    print("Starting plotting...")
    
    # Get chromosomes to plot
    if target_chrom is not None:
        # Filter to specific chromosome
        target_df = df[df["Chromosome"] == target_chrom].copy()
        if len(target_df) == 0:
            raise ValueError(f"No data found for chromosome {target_chrom}")
        chromosome_data = {target_chrom: target_df}
        print(f"Plotting single chromosome: {target_chrom}")
    else:
        # Group by chromosome
        chromosome_data = {}
        sorted_chroms = sorted(df["Chromosome"].unique(), key=chrom_sort_key)
        for chrom in sorted_chroms:
            chromosome_data[chrom] = df[df["Chromosome"] == chrom].copy()
        print(f"Plotting all chromosomes: {sorted_chroms}")
    
    # Apply filters and process each chromosome
    for chrom in list(chromosome_data.keys()):
        chrom_df = chromosome_data[chrom]
        
        # Apply MB range filter if specified
        if mb_start is not None or mb_end is not None:
            chrom_df["Mb"] = chrom_df["Start"] / 1_000_000
            
            if mb_start is not None:
                chrom_df = chrom_df[chrom_df["Mb"] >= mb_start]
                print(f"Filtered {chrom} to >= {mb_start} Mb: {len(chrom_df)} rows")
            
            if mb_end is not None:
                chrom_df = chrom_df[chrom_df["Mb"] <= mb_end]
                print(f"Filtered {chrom} to <= {mb_end} Mb: {len(chrom_df)} rows")
            
            if len(chrom_df) == 0:
                print(f"No data found in the specified range for {chrom}, skipping...")
                del chromosome_data[chrom]
                continue
        
        # Subsample positions for plotting
        print(f"Subsampling every {position_stride} positions for {chrom}...")
        unique_positions = sorted(chrom_df["Start"].unique())
        selected_positions = unique_positions[::position_stride]
        plot_df = chrom_df[chrom_df["Start"].isin(selected_positions)].reset_index(drop=True)
        
        print(f"Selected {len(plot_df)} positions for plotting {chrom}")
        chromosome_data[chrom] = plot_df
    
    # Filter out chromosomes with no data
    chromosome_data = {k: v for k, v in chromosome_data.items() if len(v) > 0}
    
    if len(chromosome_data) == 0:
        raise ValueError("No data found for any chromosome after filtering")
    
    # Create subplots
    n_chroms = len(chromosome_data)
    fig, axes = plt.subplots(n_chroms, 1, figsize=(figsize[0], figsize[1] * n_chroms))
    
    # Handle case where there's only one chromosome
    if n_chroms == 1:
        axes = [axes]
    
    # Plot each chromosome
    for idx, (chrom, plot_df) in enumerate(chromosome_data.items()):
        ax = axes[idx]
        
        # Convert to Mb for plotting
        plot_df_mb = plot_df.copy()
        plot_df_mb["Start_Mb"] = plot_df_mb["Start"] / 1_000_000
        
        # Plot the line
        sns.lineplot(
            x=plot_df_mb["Start_Mb"], 
            y=plot_df_mb["distances"], 
            hue=plot_df_mb["labels_siam"], 
            ax=ax
        )
        
        # Set x-axis limits
        x_min = plot_df_mb["Start_Mb"].min()
        x_max = plot_df_mb["Start_Mb"].max()
        
        if mb_start is not None:
            x_min = mb_start
        if mb_end is not None:
            x_max = mb_end
            
        ax.set_xlim(left=x_min, right=x_max)
        
        # Set major tick locator if specified
        if major_tick_interval is not None:
            ax.xaxis.set_major_locator(MultipleLocator(major_tick_interval))
        
        # Labels and formatting
        ax.set_xlabel("Start Position (Mb)", fontsize=16)
        ax.set_ylabel("Distance", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(title="Label", fontsize=14, title_fontsize=15)
        
        # Set title for each subplot
        if title is None:
            if distance_measure.lower() == "cosine":
                subplot_title = f"{loss_function} loss {model_name} model {chrom} distance predictions - Cosine Distance"
            else:
                subplot_title = f"{loss_function} loss {model_name} model {chrom} distance predictions - Pairwise Distance"
        else:
            subplot_title = f"{title} - {chrom}"
        
        # Add range info to title if specified
        if mb_start is not None or mb_end is not None:
            range_str = f" ({mb_start or 0:.1f}-{mb_end or x_max:.1f} Mb)"
            subplot_title += range_str
        
        ax.set_title(subplot_title, fontsize=20)
    
    plt.tight_layout()
    plt.show()
    
    print("Plot completed!")


def plot_hic_cnn_features(
    hic_files,
    paired_maps, 
    filter_features,
    model_ckpt_path,
    mlhic_dataset,
    reference_genome,
    mb_start=None,
    mb_end=None,
    distance_measure="pairwise", 
    chrom="chr2",
    bin_size=10000,
    patch_len=224,
    position_stride=10,
    chunk_size=1000,
    
    plot_w=20
):
    """
    Plot chunked panels over a genomic Mb range instead of bin count.
    
    Parameters:
    -----------
    hic_files : dict
        Mapping "KO"/"WT" to lists of (label, path)
    paired_maps : HiCDatasetDec.paired_maps
        Paired maps from HiCDatasetDec
    filter_features : list
        List of extracted CNN features
    model_ckpt_path : str
        Path to checkpoint file for trained Siamese model
    test_hic_paths : list
        Loaded mlhic files
    reference_genome : str
        Key for reference_genomes dict
    mb_start : float, optional
        Start position in Mb (inclusive). If None, starts from chromosome beginning.
    mb_end : float, optional
        End position in Mb (exclusive). If None, goes to chromosome end.
    distance_measure : str, default pairwise
        Measure for embedding distance. The options are either "cosine" or "pairwise". Supcon works best with cosine while contrastive and triplet works best with pairwise
    chrom : str
        Chromosome string, e.g. "chr2"
    bin_size : int
        Hi-C resolution in bp
    patch_len : int, default 224
        Model patch length in bins
    position_stride : int, default 10
        Subsampling stride in bins
    chunk_size : int, default 1000
        Bins per plotting chunk
    plot_w : float, default 20
        Plot width in inches
    """
    
    # Set plot style and random seed
    plt.rcParams.update({"font.size": 12})
    random.seed(100)
    
    # Get chromosome length if mb_start/mb_end not provided
    if mb_start is None or mb_end is None:
        # Open first Hi-C file to get chromosome length
        first_file = hic_files["KO"][0][1] if "KO" in hic_files else hic_files["WT"][0][1]
        hic = hicstraw.HiCFile(first_file)
        chromosomes = hic.getChromosomes()
        
        # Find the target chromosome
        chrom_length = None
        for c in chromosomes:
            if c.name == chrom:
                chrom_length = c.length
                break
        
        if chrom_length is None:
            raise ValueError(f"Chromosome {chrom} not found in Hi-C file")
        
        # Set default range if not provided
        if mb_start is None:
            mb_start = 0.0
        if mb_end is None:
            mb_end = chrom_length / 1e6  # Convert bp to Mb
        
        print(f"Auto-detected chromosome {chrom} length: {chrom_length:,} bp ({mb_end:.1f} Mb)")
    
    # Convert MB range to genomic coordinates and bins
    start = int(mb_start * 1e6)
    end = int(mb_end * 1e6) 
    start_bin = start // bin_size
    end_bin = end // bin_size
    num_bins = end_bin - start_bin
    mb_range = mb_end - mb_start
    
    print(f"Analyzing {chrom}:{mb_start:.1f}-{mb_end:.1f} Mb")
    print(f"Bin range: {start_bin}-{end_bin} ({num_bins} bins)")
    
    # Create mask
    mask = np.tril(np.ones((num_bins, num_bins)), k=-6)
    
    def process_matrix(matrix, target_width):
        """Process Hi-C matrix with rotation and cropping"""
        matrix *= mask
        matrix_rot = rotate(matrix, angle=45, reshape=True, order=1)
        half_height = matrix_rot.shape[0] // 2
        matrix_cropped = matrix_rot[half_height:, :]
        if matrix_cropped.shape[1] != target_width:
            matrix_resized = resize(
                matrix_cropped,
                (matrix_cropped.shape[0], target_width),
                order=1,
                preserve_range=True
            )
        else:
            matrix_resized = matrix_cropped
        return matrix_resized

    def load_and_sum_replicates(file_list, target_width):
        """Load and sum Hi-C replicates"""
        summed = None
        for label, path in file_list:
            print(f"  Loading {label}...")
            hic = hicstraw.HiCFile(path)
            mzd = hic.getMatrixZoomData(chrom, chrom, "observed", "KR", "BP", bin_size)
            matrix = mzd.getRecordsAsMatrix(start, end - bin_size, start, end - bin_size)
            proc = process_matrix(matrix, target_width)
            summed = proc.copy() if summed is None else summed + proc
        return summed

    # Load and process Hi-C data
    print("Processing Hi-C data...")
    print("  KO condition:")
    ko_sum = load_and_sum_replicates(hic_files["KO"], num_bins)
    print("  WT condition:")
    wt_sum = load_and_sum_replicates(hic_files["WT"], num_bins)
    diff_matrix = ko_sum-wt_sum
    print(f"  Difference matrix shape: {diff_matrix.shape}")

    # CNN CONDITION MAP EXTRACTION
    def plot_conditions_map(paired_maps, chromosome_num, bin_start=0, bin_end=None, row_end=None):
        """Extract CNN condition map"""
        all_maps = paired_maps.get(chromosome_num)
        if all_maps is None:
            raise ValueError(f"No paired maps found for chromosome {chromosome_num}")
        cond_map = all_maps["conditions"]
        cond_sum = np.nansum(cond_map, axis=0)
        if bin_end is None: 
            bin_end = cond_sum.shape[1]
        if row_end is None: 
            row_end = cond_sum.shape[0]
        half_row = cond_sum.shape[0] // 2
        return cond_sum[half_row:row_end, bin_start:bin_end]
    
    # Extract chromosome number for paired_maps lookup
    chrom_num = chrom.replace("chr", "")
    
    print("Extracting CNN condition map...")
    cnn_cond_map = plot_conditions_map(
        paired_maps=paired_maps,
        chromosome_num=chrom_num,
        bin_start=start_bin,
        bin_end=start_bin + num_bins
    )
    print(f"CNN condition map shape: {cnn_cond_map.shape}")
    
    # Align difference matrix with CNN map
    icom_diff = diff_matrix[:cnn_cond_map.shape[0], :cnn_cond_map.shape[1]]
    print(f"Aligned difference matrix shape: {icom_diff.shape}")

    # MODEL DISTANCE CALCULATIONS
    def test_model(model, dataloader, patch_len=224):
        """Test Siamese model and return distances"""
        all_distances = []
        all_labels = []
        for _, (x1, x2, y) in enumerate(dataloader):
            y = y.float()
            with torch.no_grad():
                o1, o2 = model(x1, x2)
                if distance_measure.lower() == "cosine":     
                    dists = 1 - F.cosine_similarity(o1, o2)
                elif distance_measure.lower() == "pairwise":
                    dists = F.pairwise_distance(o1, o2)
            # Expand each distance/label for all bins in the patch
            expanded_dists = np.repeat(dists.cpu().numpy(), patch_len)
            expanded_labels = np.repeat(y.cpu().numpy(), patch_len)
            all_distances.append(expanded_dists)
            all_labels.append(expanded_labels)
        return np.concatenate(all_distances), np.concatenate(all_labels)

    # Load and test model
    print("Loading Siamese model...")
    model = models.SLeNet(mask=True)
    model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
    model.eval()
    
    print("Loading test datasets...")
    siamese = SiameseHiCDataset(mlhic_dataset, reference=reference_genome)
    def normalize_chrom(chrom):
        """Strip 'chr' if present to normalize chromosome names."""
        return chrom[3:] if chrom.startswith("chr") else chrom
    
    def subset_by_chromosome(dataset, target_chrom):
        # Normalize the target too
        target_chrom = normalize_chrom(target_chrom)
        indices = [
            i for i, (_, chrom) in enumerate(dataset.pos)
            if normalize_chrom(chrom) == target_chrom
        ]
        print(f"Found {len(indices)} samples for chromosome {target_chrom}")
        return Subset(dataset, indices)
    siamese_subset = subset_by_chromosome(siamese, chrom)
    dl = DataLoader(siamese_subset, batch_size=100, sampler=SequentialSampler(siamese_subset))

    print("Running model testing...")
    distances, labels_siam = test_model(model, dl, patch_len=patch_len)

    # Get genomic positions and expand them
    print("Processing genomic positions...")
    subset_indices = siamese_subset.indices  # Get original dataset indices

    # Use original dataset method
    pd_dict = siamese.get_genomic_positions(append="chr")
    
    # Now filter it with the chromosome-specific indices
    subset_pd_dict = {
        key: [val[i] for i in subset_indices]
        for key, val in pd_dict.items()
    }
    start_positions = subset_pd_dict["Start"]
    chromosomes = subset_pd_dict["Chromosome"]
    labels = siamese.labels

    # Sanity check
    assert len(start_positions) == len(distances) // patch_len, "Mismatch in number of patches"

    # Expand genomic positions
    expanded_starts = []
    expanded_chroms = []
    expanded_true_labels = []
    for chrom_val, start_pos, label in zip(chromosomes, start_positions, labels):
        for i in range(patch_len):
            expanded_starts.append(start_pos + i * bin_size)
            expanded_chroms.append(chrom_val)
            expanded_true_labels.append(label)

    # Construct DataFrame
    pos_df = pd.DataFrame({
        "Chromosome": expanded_chroms,
        "Start": expanded_starts,
        "distance": distances,
        "labels": expanded_true_labels,
        "label_siam": np.where(labels_siam==0, "within", "between")
    })

    # Filter for target chromosome and range
    pos_df = pos_df[pos_df["Chromosome"] == chrom]
    pos_df["bin_idx"] = ((pos_df["Start"] / bin_size) - start_bin).astype(int)
    pos_df = pos_df[(pos_df["bin_idx"] >= 0) & (pos_df["bin_idx"] < num_bins)]
    pos_df["Mb"] = pos_df["Start"] / 1e6

    # Subsample every nth bin for plotting
    print("Subsampling positions for plotting...")
    unique_positions = sorted(pos_df["Start"].unique())
    selected_positions = unique_positions[::position_stride]
    pos_df = pos_df[pos_df["Start"].isin(selected_positions)].reset_index(drop=True)
    print(f"Selected {len(pos_df)} positions for plotting")

    def get_xlabels(start_bin_chunk, end_bin_chunk, total_bins, start_mb, mb_range):
        """Generate x-axis labels for genomic coordinates"""
        start_mb_chunk = start_mb + (mb_range * start_bin_chunk / total_bins)
        end_mb_chunk = start_mb + (mb_range * end_bin_chunk / total_bins)
        tick_mbs = np.arange(np.ceil(start_mb_chunk * 2) / 2, end_mb_chunk, 0.5)
        tick_bins = ((tick_mbs - start_mb) / mb_range) * total_bins - start_bin_chunk
        return tick_bins, [f"{mb:.1f}" for mb in tick_mbs]

    def reconstruct_cnn_features(features, paired_maps, chromosome_num, pixel_size, num_bins):
        """Reconstruct CNN features canvas"""
        grouped = paired_maps.get(chromosome_num, None)
        if grouped is None:
            raise RuntimeError(f"No CNN maps for chromosome {chromosome_num}")
        rep_map = grouped["replicate"]
        canvas_bins = len(rep_map[0][0])
        canvas = np.zeros((pixel_size, canvas_bins), dtype=np.float32)
        
        for feat in features:
            idx, patch, orig_dims, y0, count, pos_or_neg, score, (chrom_feat, i0, i1, j0, j1) = feat
            if chrom_feat != chromosome_num:
                continue
            patch_up = resize(
                patch.astype(float), output_shape=orig_dims,
                order=1, preserve_range=True, anti_aliasing=False
            )
            yslice = slice(i0, i0 + orig_dims[0])
            xslice = slice(j0, j0 + orig_dims[1])
            if yslice.stop > pixel_size or xslice.stop > canvas_bins:
                continue
            canvas[yslice, xslice] += (patch_up if pos_or_neg == 0 else -patch_up)
        return canvas

    # Reconstruct CNN features
    print("Reconstructing CNN features...")
    height, width = cnn_cond_map.shape
    actual_height = icom_diff.shape[0]
    
    cnn_feats = reconstruct_cnn_features(
        features=filter_features,
        paired_maps=paired_maps,
        chromosome_num=chrom_num,
        pixel_size=actual_height,
        num_bins=width
    )
    print(f"CNN features canvas shape: {cnn_feats.shape}")

    # Calculate plot dimensions
    plot_height_per_plot = plot_w * (actual_height / chunk_size)
    fig_height = 5 * plot_height_per_plot  # 5 subplots
    
    dist_min, dist_max = pos_df["distance"].min(), pos_df["distance"].max()

    # PLOTTING
    print("Starting plotting...")
    plot_count = 0
    
    for bs in range(0, width, chunk_size):
        be = min(bs + chunk_size, width)
        w = be - bs
        if w <= 0:
            continue

        # Calculate Mb range for this chunk
        mb_s = mb_start + mb_range * bs / width
        mb_e = mb_start + mb_range * be / width

        print(f"Plotting chunk {plot_count + 1}: {mb_s:.1f}-{mb_e:.1f} Mb")

        # Extract data chunks
        cnn_chunk = cnn_cond_map[:actual_height, bs:be]
        hic_chunk = icom_diff[:actual_height, bs:be]
        
        # Extract feature chunk (adjust indexing based on your feature coordinate system)
        feat_start_idx = int(mb_s * 100)  # Adjust multiplier as needed
        feat_end_idx = int(mb_e * 100)
        if feat_end_idx <= cnn_feats.shape[1]:
            feat_chunk = cnn_feats[:, feat_start_idx:feat_end_idx]
        else:
            # Pad or truncate as needed
            feat_chunk = np.zeros((actual_height, w))
            available_width = min(w, cnn_feats.shape[1] - feat_start_idx)
            if available_width > 0:
                feat_chunk[:, :available_width] = cnn_feats[:, feat_start_idx:feat_start_idx + available_width]

        # Filter position data for this chunk
        subdf = pos_df[(pos_df["bin_idx"] >= bs) & (pos_df["bin_idx"] < be)].copy()
        subdf["x_bin"] = subdf["bin_idx"] - bs

        # Create 5 subplots - Remove sharex=True to allow independent x-axis labeling
        fig, axs = plt.subplots(
            5, 1,
            figsize=(plot_w, fig_height),
            constrained_layout=True
        )
        
        xticks, xtick_labels = get_xlabels(bs, be, width, mb_start, mb_range)

        # 1) CNN Map
        im0 = axs[0].imshow(
            cnn_chunk, cmap='bwr', origin='lower', aspect='auto',
            vmin=-np.nanmax(np.abs(cnn_cond_map)),
            vmax=np.nanmax(np.abs(cnn_cond_map)),
            extent=[0, w, 0, actual_height]
        )
        axs[0].set_title(f"CNN Map: {mb_s:.1f}–{mb_e:.1f} Mb")
        axs[0].set_ylabel("Map Row")
        axs[0].set_ylim(0, actual_height)
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels(xtick_labels, rotation=45)
        axs[0].set_xlabel("Genomic Position (Mb)")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        # 2) Hi‑C Δ
        im1 = axs[1].imshow(
            hic_chunk, cmap='RdBu_r', origin='lower', aspect='auto',
            vmin=-np.percentile(np.abs(icom_diff), 99),
            vmax=np.percentile(np.abs(icom_diff), 99),
            extent=[0, w, 0, actual_height]
        )
        axs[1].set_title(f"Hi‑C Δ: {mb_s:.1f}–{mb_e:.1f} Mb")
        axs[1].set_ylabel("Map Row")
        axs[1].set_ylim(0, actual_height)
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(xtick_labels, rotation=45)
        axs[1].set_xlabel("Genomic Position (Mb)")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # 3) Extracted CNN Features
        im_feat = axs[2].imshow(
            feat_chunk, cmap='bwr', origin='lower', aspect='auto',
            vmin=-np.nanmax(np.abs(cnn_feats)) if np.nanmax(np.abs(cnn_feats)) > 0 else -1,
            vmax=np.nanmax(np.abs(cnn_feats)) if np.nanmax(np.abs(cnn_feats)) > 0 else 1,
            extent=[0, w, 0, actual_height]
        )
        axs[2].set_title(f"Extracted CNN Features: {mb_s:.1f}–{mb_e:.1f} Mb")
        axs[2].set_ylabel("Map Row")
        axs[2].set_ylim(0, actual_height)
        axs[2].set_xticks(xticks)
        axs[2].set_xticklabels(xtick_labels, rotation=45)
        axs[2].set_xlabel("Genomic Position (Mb)")
        plt.colorbar(im_feat, ax=axs[2], fraction=0.046, pad=0.04)

        # 4) Overlay
        axs[3].imshow(
            hic_chunk, cmap='RdBu_r', origin='lower', aspect='auto',
            vmin=-np.percentile(np.abs(icom_diff), 99),
            vmax=np.percentile(np.abs(icom_diff), 99),
            extent=[0, w, 0, actual_height], alpha=1.0
        )
        im2 = axs[3].imshow(
            cnn_chunk, cmap='bwr', origin='lower', aspect='auto',
            vmin=-np.nanmax(np.abs(cnn_cond_map)),
            vmax=np.nanmax(np.abs(cnn_cond_map)),
            extent=[0, w, 0, actual_height], alpha=0.6
        )
        axs[3].set_title(f"Overlay: {mb_s:.1f}–{mb_e:.1f} Mb")
        axs[3].set_ylabel("Map Row")
        axs[3].set_ylim(0, actual_height)
        axs[3].set_xticks(xticks)
        axs[3].set_xticklabels(xtick_labels, rotation=45)
        axs[3].set_xlabel("Genomic Position (Mb)")
        plt.colorbar(im2, ax=axs[3], fraction=0.046, pad=0.04)

        # 5) Pairwise Distances
        if len(subdf) > 0:
            sns.lineplot(
                data=subdf, x="x_bin", y="distance", hue="label_siam",
                ax=axs[4]
            )
        if distance_measure.lower() == "cosine":     
            axs[4].set_title(f"1-Cosine Similarity Embedding Distances: {mb_s:.1f}–{mb_e:.1f} Mb")
        elif distance_measure.lower() == "pairwise":
            axs[4].set_title(f"Pairwise Similarity Embedding Distances: {mb_s:.1f}–{mb_e:.1f} Mb")
        axs[4].set_xlabel("Genomic Position (Mb)")
        axs[4].set_ylabel("Distance")
        axs[4].set_xticks(xticks)
        axs[4].set_xticklabels(xtick_labels, rotation=45)
        axs[4].set_xlim(0, w)
        axs[4].set_ylim(dist_min, dist_max)
        axs[4].legend(title="Label", fontsize=12, title_fontsize=13, loc="upper right")

        # Set x-axis limits for all subplots
        for ax in axs:
            ax.set_xlim(0, w)

        plt.show()
        plot_count += 1
    
    print(f"Completed plotting {plot_count} chunks!")