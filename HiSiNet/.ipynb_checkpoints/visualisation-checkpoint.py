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
from scipy.stats import gaussian_kde
from scipy import integrate
from matplotlib.colors import Normalize, TwoSlopeNorm


def plot_chromosome_distances_middle_bin(
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
    font_size=14,              # NEW: global font size
    show_difference_plot=True, # NEW: add difference plot
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
        Subsampling stride for plotting. Want 10 if showing all chrom at once or if plotting on the fly
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
        model = model_class(mask=True, image_size=patch_len)
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
        def test_model(model, dataloader):
            """Test Siamese model and return distances - ONE DISTANCE PER TILE"""
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
                
                # KEY CHANGE: No expansion - one distance per tile
                all_distances.append(dists.cpu().numpy())
                all_labels.append(y.cpu().numpy())
            
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
            distances, labels_siam = test_model(model, dataloader)
            
            print("Processing genomic positions...")
            
            # --- Get genomic positions ---
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
            
            # KEY CHANGE: Use middle bin positions instead of expanding
            middle_positions = [pos + (patch_len * bin_size) // 2 for pos in start_positions]
            
            # Sanity check - now we should have 1:1 correspondence
            assert len(middle_positions) == len(distances), f"Mismatch: {len(middle_positions)} positions vs {len(distances)} distances"
            
            # --- Construct DataFrame ---
            chrom_df = pd.DataFrame({
                "Chromosome": chromosomes,
                "Middle": middle_positions,  # Using middle bin positions
                "distances": distances,
                "labels": subset_labels,
                "labels_siam": np.where(labels_siam==0, "within", "between")
            })
            
            print(f"Created DataFrame with {len(chrom_df)} rows for {chrom} (middle bin positions)")
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
        unique_positions = sorted(chrom_df["Middle"].unique())
        selected_positions = unique_positions[::position_stride]
        plot_df = chrom_df[chrom_df["Middle"].isin(selected_positions)].reset_index(drop=True)
        
        print(f"Selected {len(plot_df)} positions for plotting {chrom}")
        chromosome_data[chrom] = plot_df
    
    # Filter out chromosomes with no data
    chromosome_data = {k: v for k, v in chromosome_data.items() if len(v) > 0}
    
    if len(chromosome_data) == 0:
        raise ValueError("No data found for any chromosome after filtering")
    
    # Create subplots
    n_chroms = len(chromosome_data)
    n_rows = n_chroms * (2 if show_difference_plot else 1)
    
    fig, axes = plt.subplots(
        n_rows, 1, 
        figsize=(figsize[0], figsize[1] * n_rows),
        sharex=False
    )
    
    if n_rows == 1:
        axes = [axes]  # make iterable
    
    # Plot each chromosome
    for idx, (chrom, plot_df) in enumerate(chromosome_data.items()):
        ax_main = axes[idx * (2 if show_difference_plot else 1)]
        
        # Convert to Mb for plotting
        plot_df_mb = plot_df.copy()
        plot_df_mb["Middle_Mb"] = plot_df_mb["Middle"] / 1_000_000
        
        # Main plot: within vs between
        sns.lineplot(
            x=plot_df_mb["Middle_Mb"], 
            y=plot_df_mb["distances"], 
            hue=plot_df_mb["labels_siam"], 
            ax=ax_main
        )
        ax_main.legend_.set_title(None)
        
        # Set labels and fonts
        ax_main.set_xlabel("Position (Mb)", fontsize=font_size)
        ax_main.set_ylabel("Distance", fontsize=font_size)
        ax_main.tick_params(axis='both', labelsize=font_size-2)
        ax_main.set_xlim(plot_df_mb["Middle_Mb"].min(), plot_df_mb["Middle_Mb"].max())  # << fix)

        # Set major ticks based on interval in Mb
        if major_tick_interval is not None:
            ax_main.xaxis.set_major_locator(MultipleLocator(major_tick_interval))
        
        # Title
        if title is None:
            if distance_measure.lower() == "cosine":
                subplot_title = f"{loss_function} loss {model_name} {chrom} - Cosine (Middle Bin)"
            else:
                subplot_title = f"{loss_function} loss {model_name} {chrom} - Pairwise (Middle Bin)"
        else:
            subplot_title = f"{title}"
        
        ax_main.set_title(subplot_title, fontsize=font_size+2)

# --- difference plot ---
        if show_difference_plot:
            ax_diff = axes[idx*2 + 1]
            mean_df = (
                plot_df_mb.groupby("Middle_Mb")
                .apply(lambda g: g[g["labels_siam"]=="between"]["distances"].mean()
                                - g[g["labels_siam"]=="within"]["distances"].mean())
                .reset_index(name="delta_dist")
            )
            mean_df["delta_dist"] = mean_df["delta_dist"].clip(lower=0)
        
            sns.lineplot(x="Middle_Mb", y="delta_dist", data=mean_df, color="black", ax=ax_diff)
            ax_diff.set_ylim(bottom=0)
            ax_diff.set_xlim(mean_df["Start_Mb"].min(), mean_df["Start_Mb"].max())  # << fix
            if major_tick_interval is not None:
                ax_diff.xaxis.set_major_locator(MultipleLocator(major_tick_interval))
            ax_diff.set_xlabel("Position (Mb)", fontsize=font_size)
            ax_diff.set_ylabel("Δ Distance\n(condition - replicate)", fontsize=font_size)
            ax_diff.tick_params(axis='both', labelsize=font_size-2)
            ax_diff.set_title(f"{chrom} Twins Score", fontsize=font_size+1)
    
    plt.tight_layout()
    plt.show()

def plot_smoothed_distance_hist(
    df,
    distance_col="distances",
    label_col="labels_siam",
    within_label="within",
    between_label="between",
    legend_labels=("Within condition", "Between conditions"),  # <- new
    bins=200,
    bandwidth=None,
    figsize=(6, 4.5),
    colors=("#418BBF", "#FF7E0C"),
    alpha=0.5,
    title="Test Chromosome Condition vs Replicate Pair Distance Distributions",
    fontsize=14  
):
    """
    Plot smoothed KDEs of distance distributions and compute performance metrics.

    Returns
    -------
    float or None
        Intersection threshold, or None if not found.
    """
    # Extract data
    d_w = df.loc[df[label_col] == within_label, distance_col].values
    d_b = df.loc[df[label_col] == between_label, distance_col].values
    if len(d_w) == 0 or len(d_b) == 0:
        raise ValueError("No data for one of the labels")

    # KDE grid
    mn = min(d_w.min(), d_b.min())
    mx = max(d_w.max(), d_b.max())
    x = np.linspace(mn, mx, bins)

    # KDEs
    kde_w = gaussian_kde(d_w, bw_method=bandwidth)
    kde_b = gaussian_kde(d_b, bw_method=bandwidth)
    y_w = kde_w(x)
    y_b = kde_b(x)

    # Intersection point (threshold)
    diff = y_w - y_b
    idx = np.where(np.diff(np.sign(diff)))[0]
    if idx.size:
        i = idx[0]
        x0, x1 = x[i], x[i+1]
        d0, d1 = diff[i], diff[i+1]
        intersect = x0 - d0 * (x1 - x0) / (d1 - d0)
    else:
        intersect = None

    # Compute metrics if threshold found
    replicate_rate = condition_rate = mean_perf = separation = None
    if intersect is not None:
        replicate_rate = (d_w < intersect).mean()
        condition_rate = (d_b > intersect).mean()
        mean_perf = (replicate_rate + condition_rate) / 2

        # Separation index (1 - overlapping area)
        overlap = np.minimum(y_w, y_b)
        area_overlap = integrate.simps(overlap, x)
        separation = 1 - area_overlap

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x, y_w, color=colors[0], alpha=alpha, label=legend_labels[0])
    ax.fill_between(x, y_b, color=colors[1], alpha=alpha, label=legend_labels[1])

    ax.set_title(title, fontsize=fontsize+2)
    ax.set_xlabel("Embedding Distance", fontsize=fontsize)
    ax.set_ylabel("Probability Density", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize-2)
    ax.legend(fontsize=fontsize-2)
    plt.tight_layout()
    plt.show()

    # Output results
    print(f"Intersection threshold: {intersect}")
    if intersect is not None:
        print(f"Replicate rate:  {replicate_rate:.2%}")
        print(f"Condition rate:  {condition_rate:.2%}")
        print(f"Mean performance: {mean_perf:.2%}")
        print(f"Separation index: {separation:.2%}")

    return intersect

def plot_hic_region(
    hic_files: dict,
    chrom: str,
    start_mb: float,
    end_mb: float,
    bin_size: int = 10000,
    norm: str = "KR",
    matrix_type: str = "observed",
    cmap="Reds",
    vmin_pct=0,
    vmax_pct=99,
    figsize=(10, 10),
    mb_per_tick: float = 0.2
):
    """
    Plot a 2×2 grid of Hi‑C contact maps for WT/KO replicates over a genomic interval,
    each subplot with its own ticks and rotated labels.
    """
    # Convert Mb to base pairs
    start = int(start_mb * 1e6)
    end   = int(end_mb   * 1e6)
    
    # Load and symmetrize all four matrices
    mats, titles = [], []
    for cond in ("WT", "KO"):
        for label, path in hic_files[cond]:
            hf  = hicstraw.HiCFile(path)
            mzd = hf.getMatrixZoomData(chrom, chrom, matrix_type, norm, "BP", bin_size)
            mat = mzd.getRecordsAsMatrix(start, end-bin_size, start, end-bin_size)
            mat = np.nan_to_num(mat)
            mats.append(mat)
            titles.append(f"{cond} {label}")

    # Compute shared color limits
    all_vals = np.concatenate([m.flatten() for m in mats])
    vmin = np.percentile(all_vals, vmin_pct)
    vmax = np.percentile(all_vals, vmax_pct)

    # Create tick values
    n_bins = (end - start) // bin_size
    tick_step = int((mb_per_tick * 1e6) // bin_size)
    tick_indices = np.arange(0, n_bins, tick_step)
    tick_labels = (start_mb + tick_indices * bin_size / 1e6).round(2)

    # Create 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize,
                             constrained_layout=True)
    axes = axes.flatten()

    # Plot each matrix
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(
            mat, #np.tril(mat, k=-3) + np.triu(mat, k=3),  # hide diagonal ±3
            cmap=cmap,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            extent=[start_mb, end_mb, end_mb, start_mb],
            aspect="equal"
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticks(tick_labels)
        ax.set_yticks(tick_labels)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticklabels(tick_labels, rotation=45, va='top')
        ax.set_xlabel("Mb")
        ax.set_ylabel("Mb")

    # Single colorbar
    cbar = fig.colorbar(im, ax=axes.tolist(),
                        orientation="vertical",
                        fraction=0.02, pad=0.04)
    cbar.set_label("Contact frequency", rotation=90)

    plt.suptitle(f"Hi-C contact maps: {chrom} {start_mb}-{end_mb} Mb", fontsize=14)
    plt.show()

def analyze_and_plot_hic_tiles(
    model,
    hic_files: dict,
    chrom: str,
    start_mb: float = None,
    middle_mb: float = None,
    tile1_file: str = None,
    tile2_file: str = None,
    bin_size: int = 10000,
    norm: str = "KR",
    matrix_type: str = "observed",
    tile_size_mb: float = 2.24,
    distance_measure: str = "pairwise",
    n_steps: int = 100
):
    """
    Extract Hi-C tiles, compare them using a trained model, and plot the results.

    Parameters
    ----------
    start_mb : float, optional
        Starting position in Mb. Mutually exclusive with middle_mb.
    middle_mb : float, optional
        Midpoint position in Mb. Mutually exclusive with start_mb.
    """

    # ===== COORDINATE LOGIC =====
    if (start_mb is None) == (middle_mb is None):
        raise ValueError("You must provide exactly one of start_mb or middle_mb")

    if middle_mb is not None:
        start_mb = middle_mb - tile_size_mb / 2
        end_mb = middle_mb + tile_size_mb / 2
    else:
        end_mb = start_mb + tile_size_mb

    tile_start = int(start_mb * 1e6)
    tile_end = int(end_mb * 1e6) - bin_size
    
    # ===== EXTRACTION FUNCTIONS =====
    def extract_tile(file_name):
        """Extract a single Hi-C tile"""
        # Find the correct file path by searching through all conditions
        file_path = None
        condition_found = None
        
        for condition in hic_files:
            for label, path in hic_files[condition]:
                if label == file_name:
                    file_path = path
                    condition_found = condition
                    break
            if file_path:
                break
        
        if file_path is None:
            available_files = []
            for condition in hic_files:
                for label, path in hic_files[condition]:
                    available_files.append(label)
            raise ValueError(f"Could not find file '{file_name}'. Available files: {available_files}")
        
        print(f"Using {condition_found} file: {file_name}")
        
        # Load Hi-C data
        hf = hicstraw.HiCFile(file_path)
        mzd = hf.getMatrixZoomData(chrom, chrom, matrix_type, norm, "BP", bin_size)
        mat = mzd.getRecordsAsMatrix(tile_start, tile_end, tile_start, tile_end)
        
        # Clean and symmetrize
        mat = np.nan_to_num(mat)
        
        return mat
    
    def distance_fn(i1, i2):
        """Distance function for saliency computation"""
        o1 = model.forward_one(i1)
        o2 = model.forward_one(i2)
        if distance_measure.lower() == "cosine":
            dists = 1 - F.cosine_similarity(o1, o2)
        elif distance_measure.lower() == "pairwise":
            dists = F.pairwise_distance(o1, o2)
        else:
            raise ValueError(f"Unknown distance measure: {distance_measure}")
        return dists
    
    def integrated_gradients(input_tensor, baseline_tensor, other_tensor, n_steps=50, device="cpu"):
        """Compute integrated gradients for saliency"""
        input_t = input_tensor.clone().detach().to(device).requires_grad_(True)
        baseline_t = baseline_tensor.to(device)
        total_grad = torch.zeros_like(input_t)

        for alpha in np.linspace(0.0, 1.0, n_steps, endpoint=True):
            interp = baseline_t + alpha * (input_t - baseline_t)
            # choose which side is varying
            if other_tensor is input_tensor:
                dist = distance_fn(baseline_t, interp)
            else:
                dist = distance_fn(interp, other_tensor)
            dist.backward()
            total_grad += input_t.grad
            input_t.grad.zero_()

        avg_grad = total_grad / n_steps
        attributions = (input_t - baseline_t) * avg_grad
        return attributions.detach().squeeze().cpu().numpy()
    
    def get_sym_norm(arr, low_pct=1, high_pct=99):
        """Get symmetric normalization for plotting"""
        lo, hi = np.percentile(arr, [low_pct, high_pct])
        m = max(abs(lo), abs(hi))
        return TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
    
    # ===== TILE EXTRACTION =====
    print(f"Extracting tile 1: {tile1_file}")
    tile1 = extract_tile(tile1_file)
    
    print(f"Extracting tile 2: {tile2_file}")
    tile2 = extract_tile(tile2_file)
    
    # Convert to torch tensors and add batch/channel dimensions
    x1 = torch.from_numpy(tile1).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]
    x2 = torch.from_numpy(tile2).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]
    x1_norm = x1 / x1.amax(dim=(1, 2, 3), keepdim=True)
    x2_norm = x2 / x2.amax(dim=(1, 2, 3), keepdim=True)
    
    # Apply mask (remove diagonal ±3)
    mask = np.tril(np.ones(int(tile_size_mb*100)), k=-3) + np.triu(np.ones(int(tile_size_mb*100)), k=3)
    mask_torch = torch.from_numpy(mask).float().to(x1.device).unsqueeze(0).unsqueeze(0)
    x1_masked = x1_norm * mask_torch
    x2_masked = x2_norm * mask_torch
    print(f"Embedding Distance: {round(distance_fn(x1_masked, x2_masked).item(), 4)}")
    
    # ===== FEATURE EXTRACTION =====
    feature_results = None
    try:
        activation = {}
        def get_activation(name):
            def hook(model, inp, out):
                activation[name] = out.detach()
            return hook
        
        hook = model.features[-2].register_forward_hook(get_activation('feat'))
        
        # Extract features
        model.eval()
        with torch.no_grad():
            _ = model.forward_one(x1_masked)
            feat_x1 = activation['feat'].squeeze(0)
            _ = model.forward_one(x2_masked)
            feat_x2 = activation['feat'].squeeze(0)
        
        hook.remove()
        
        # Average across channels
        avg_feat_x1 = feat_x1.mean(dim=0).cpu().numpy()
        avg_feat_x2 = feat_x2.mean(dim=0).cpu().numpy()
        diff_map = np.abs(avg_feat_x2) - np.abs(avg_feat_x1)
        
        feature_results = {
            'x1': avg_feat_x1, 
            'x2': avg_feat_x2, 
            'diff': diff_map
        }
        print("Features extracted successfully")
        
    except (AttributeError, Exception) as e:
        print(f"Feature extraction not available: {e}")
        feature_results = None
    
    # Prepare images for plotting
    img_x1 = (x1 * mask_torch).squeeze().cpu().numpy()
    img_x2 = (x2 * mask_torch).squeeze().cpu().numpy()
    
    # ===== SALIENCY COMPUTATION =====
    saliency_results = None
    try:
        print("Computing saliency maps...")
        # Compute saliency maps
        attr_x1 = integrated_gradients(x1_masked, x2_masked, x2_masked, n_steps=n_steps)
        attr_x2 = integrated_gradients(x2_masked, x1_masked, x1_masked, n_steps=n_steps)
        
        # Combined saliency
        combined_saliency = 0.5 * (attr_x1 + attr_x2)
        
        # Hi-C difference map
        hic_diff = img_x2 - img_x1
        
        saliency_results = {
            'attr_x1': attr_x1,
            'attr_x2': attr_x2,
            'combined': combined_saliency,
            'hic_diff': hic_diff
        }
        print("Saliency maps computed successfully")
        
    except Exception as e:
        print(f"Saliency computation not available: {e}")
        saliency_results = None
    
    # ===== PREPARE RESULTS =====
    results = {
        'tiles': {'x1': img_x1, 'x2': img_x2},
        'features': feature_results,
        'saliency': saliency_results,
        'files': {'tile1': tile1_file, 'tile2': tile2_file},
        'coordinates': {
            'chrom': chrom,
            'start_mb': start_mb,
            'end_mb': start_mb + tile_size_mb
        }
    }
    
    # ===== PLOTTING =====
    has_features = feature_results is not None
    has_saliency = saliency_results is not None

    # Set up normalization for original images
    vmin_orig, vmax_orig = np.percentile(
        np.concatenate([img_x1.ravel(), img_x2.ravel()]), [1, 99]
    )
    norm_orig = Normalize(vmin=vmin_orig, vmax=vmax_orig)

    # Set up normalization for features if available
    if has_features:
        avg_feat_x1 = feature_results['x1']
        avg_feat_x2 = feature_results['x2']
        diff_map = feature_results['diff']
        
        combined_feat_data = np.concatenate([avg_feat_x1.ravel(), avg_feat_x2.ravel()])
        common_feat_max = max(abs(np.percentile(combined_feat_data, [1, 99])))
        norm_feat_common = TwoSlopeNorm(vmin=-common_feat_max, vcenter=0, vmax=common_feat_max)
        norm_ft_dif = get_sym_norm(diff_map)

    # Generate coordinate labels
    coords = f"{start_mb:.2f}-{start_mb + tile_size_mb:.2f} Mb"

    # ===== PLOTTING LOGIC =====
    if not has_saliency:
        # Standard mode without saliency
        n_panels = 2 + (3 if has_features else 0)
        fig, axs = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 5))
        if n_panels == 2:
            axs = [axs[0], axs[1]]  # ensure consistent indexing

        # Plot original tiles
        axs[0].imshow(img_x1, cmap='Reds', norm=norm_orig)
        axs[0].set_title(f"Tile 1: {tile1_file}\n{coords}")
        axs[0].axis('off')
        fig.colorbar(axs[0].images[0], ax=axs[0], fraction=0.046, pad=0.04)

        axs[1].imshow(img_x2, cmap='Reds', norm=norm_orig)
        axs[1].set_title(f"Tile 2: {tile2_file}\n{coords}")
        axs[1].axis('off')
        fig.colorbar(axs[1].images[0], ax=axs[1], fraction=0.046, pad=0.04)

        # Plot features if available
        if has_features:
            axs[2].imshow(avg_feat_x1, cmap='RdBu_r', norm=norm_feat_common)
            axs[2].set_title("Avg features Tile 1")
            axs[2].axis('off')
            fig.colorbar(axs[2].images[0], ax=axs[2], fraction=0.046, pad=0.04)

            axs[3].imshow(avg_feat_x2, cmap='RdBu_r', norm=norm_feat_common)
            axs[3].set_title("Avg features Tile 2")
            axs[3].axis('off')
            fig.colorbar(axs[3].images[0], ax=axs[3], fraction=0.046, pad=0.04)

            axs[4].imshow(diff_map, cmap='RdBu_r', norm=norm_ft_dif)
            axs[4].set_title("Feature Difference (Tile 2 − Tile 1)")
            axs[4].axis('off')
            fig.colorbar(axs[4].images[0], ax=axs[4], fraction=0.046, pad=0.04)

    else:
        # Saliency plotting mode
        combined_saliency = saliency_results['combined']
        hic_diff = saliency_results['hic_diff']
        norm_hic_diff = get_sym_norm(hic_diff)
        norm_saliency = get_sym_norm(combined_saliency)

        fig, axs = plt.subplots(2, 5, figsize=(24, 10))

        # Top row - original data and features
        axs[0, 0].imshow(img_x1, cmap='Reds', norm=norm_orig)
        axs[0, 0].set_title(f"Tile 1: {tile1_file}\n{coords}")
        axs[0, 0].axis('off')
        fig.colorbar(axs[0, 0].images[0], ax=axs[0, 0], fraction=0.046, pad=0.04)

        axs[0, 1].imshow(img_x2, cmap='Reds', norm=norm_orig)
        axs[0, 1].set_title(f"Tile 2: {tile2_file}\n{coords}")
        axs[0, 1].axis('off')
        fig.colorbar(axs[0, 1].images[0], ax=axs[0, 1], fraction=0.046, pad=0.04)

        if has_features:
            axs[0, 2].imshow(avg_feat_x1, cmap='RdBu_r', norm=norm_feat_common)
            axs[0, 2].set_title("Avg features Tile 1")
            axs[0, 2].axis('off')
            fig.colorbar(axs[0, 2].images[0], ax=axs[0, 2], fraction=0.046, pad=0.04)

            axs[0, 3].imshow(avg_feat_x2, cmap='RdBu_r', norm=norm_feat_common)
            axs[0, 3].set_title("Avg features Tile 2")
            axs[0, 3].axis('off')
            fig.colorbar(axs[0, 3].images[0], ax=axs[0, 3], fraction=0.046, pad=0.04)

            axs[0, 4].imshow(diff_map, cmap='RdBu_r', norm=norm_ft_dif)
            axs[0, 4].set_title("Feature Difference (Tile 2 − Tile 1)")
            axs[0, 4].axis('off')
            fig.colorbar(axs[0, 4].images[0], ax=axs[0, 4], fraction=0.046, pad=0.04)
        else:
            # Hide unused subplots
            for i in range(2, 5):
                axs[0, i].axis('off')

        # Bottom row - saliency analysis
        axs[1, 0].imshow(hic_diff, cmap='RdBu_r', norm=norm_hic_diff)
        axs[1, 0].set_title("Hi-C Difference (x₂ − x₁)")
        axs[1, 0].axis('off')
        fig.colorbar(axs[1, 0].images[0], ax=axs[1, 0], fraction=0.046, pad=0.04)

        axs[1, 1].imshow(combined_saliency, cmap='RdBu_r', norm=norm_saliency)
        axs[1, 1].set_title("Average Saliency")
        axs[1, 1].axis('off')
        fig.colorbar(axs[1, 1].images[0], ax=axs[1, 1], fraction=0.046, pad=0.04)

        axs[1, 2].imshow(img_x1, cmap='Reds', norm=norm_orig)
        axs[1, 2].imshow(combined_saliency, cmap='RdBu_r', norm=norm_saliency, alpha=0.4)
        axs[1, 2].set_title("Tile 1 with Saliency")
        axs[1, 2].axis('off')

        axs[1, 3].imshow(img_x2, cmap='Reds', norm=norm_orig)
        axs[1, 3].imshow(combined_saliency, cmap='RdBu_r', norm=norm_saliency, alpha=0.4)
        axs[1, 3].set_title("Tile 2 with Saliency")
        axs[1, 3].axis('off')

        axs[1, 4].imshow(hic_diff, cmap='RdBu_r', norm=norm_hic_diff)
        axs[1, 4].imshow(combined_saliency, cmap='RdBu_r', norm=norm_saliency, alpha=0.4)
        axs[1, 4].set_title("Hi-C Diff with Saliency")
        axs[1, 4].axis('off')

    plt.tight_layout()
    plt.show()

def plot_hic_cnn_features(
    hic_files,
    paired_maps=None, 
    filter_features=None,
    model_ckpt_path=None,
    mlhic_dataset=None,
    reference_genome=None,
    mb_start=None,
    mb_end=None,
    distance_measure="pairwise", 
    chrom="chr2",
    model_name="SLeNet",
    bin_size=10000,
    patch_len=224,
    chunk_size=1000,
    plot_w=30,
    # New parameters for average difference calculation
    window_mb=2.24,  # Window size for average calculation in Mb
    # Enhanced parameters for multiple distance input sources
    csv_paths=None,  # List of CSV file paths
    csv_data_list=None,  # List of DataFrames
    model_ckpt_paths=None,  # List of model checkpoint paths
    distance_titles=None,  # List of titles for each distance plot
    distance_measures=None,  # List of distance measures (one per model)
    # NEW: Boolean parameters for optional features
    plot_windowed_hic_avg=True,  # Plot windowed Hi-C average line
    plot_cnn_diagonal_avg=True,  # Plot CNN diagonal average line
    plot_cnn_column_avg=True,  # Plot CNN column average line
    plot_saliency_conditions=True,  # Plot saliency conditions map
):
    """
    Plot chunked panels over a genomic Mb range instead of bin count.
    Can either calculate distances on-the-fly or use pre-calculated data from CSV.
    Now supports multiple distance plots from different sources (models, CSVs, DataFrames)
    and handles missing feature maps gracefully.
    
    
    Parameters:
    -----------
    hic_files : dict
        Mapping "KO"/"WT" to lists of (label, path)
    paired_maps : HiCDatasetDec.paired_maps, optional
        Paired maps from HiCDatasetDec. If None, CNN maps won't be plotted.
    filter_features : list, optional
        List of extracted CNN features. If None, feature maps won't be plotted.
    model_ckpt_path : str, optional
        Path to checkpoint file for trained Siamese model (legacy parameter)
    mlhic_dataset : list, optional
        Loaded mlhic files (legacy parameter)
    reference_genome : str, optional
        Key for reference_genomes dict (legacy parameter)
    mb_start : float, optional
        Start position in Mb (inclusive). If None, starts from chromosome beginning.
    mb_end : float, optional
        End position in Mb (exclusive). If None, goes to chromosome end.
    distance_measure : str, default pairwise
        Measure for embedding distance (legacy parameter). Options: "cosine" or "pairwise"
    chrom : str
        Chromosome string, e.g. "chr2"
    model_name: str
        Name of CNN model tested
    bin_size : int
        Hi-C resolution in bp
    patch_len : int, default 224
        Model patch length in bins
    chunk_size : int, default 1000
        Bins per plotting chunk
    plot_w : float, default 20
        Plot width in inches
    window_mb : float, default 5
        Window size in Mb for calculating average differences
    csv_paths : list of str, optional
        List of paths to CSV files with pre-calculated distances
    csv_data_list : list of pd.DataFrame, optional
        List of pre-calculated distances DataFrames
    model_ckpt_paths : list of str, optional
        List of paths to model checkpoint files
    distance_titles : list of str, optional
        List of titles for each distance plot. If None, uses auto-generated titles
    distance_measures : list of str, optional
        List of distance measures for each model (one per model checkpoint)
        Options: "cosine" or "pairwise". If None, uses default distance_measure for all
    plot_windowed_hic_avg : bool, default True
        Whether to plot windowed Hi-C average line plot
    plot_cnn_diagonal_avg : bool, default True
        Whether to plot CNN diagonal average line plot
    plot_cnn_column_avg : bool, default True
        Whether to plot CNN column average line plot
    plot_saliency_conditions : bool, default True
        Whether to plot saliency conditions map
    """
    
    # Set plot style and random seed
    plt.rcParams.update({"font.size": 12})
    random.seed(100)
    
    # Determine what components we have available
    has_cnn_maps = paired_maps is not None
    has_features = filter_features is not None
    has_saliency = has_cnn_maps and plot_saliency_conditions
    
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
    
    # Calculate window parameters for average difference calculation (only if needed)
    windowed_avg = None
    if plot_windowed_hic_avg:
        W = int((window_mb * 1e6) / bin_size)  # Window size in bins
        total_bins = int(chrom_length / bin_size) if 'chrom_length' in locals() else end_bin + W
        ext_start = max(start_bin - W, 0)
        ext_end = min(end_bin + W, total_bins - 1)
        
        print(f"Extended region for windowed average: {ext_start}-{ext_end} bins (±{window_mb} Mb window)")
    
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

    def load_and_sum_replicates(file_list, start_coord, end_coord, target_width):
        """Load and sum Hi-C replicates"""
        summed = None
        for label, path in file_list:
            print(f"  Loading {label}...")
            hic = hicstraw.HiCFile(path)
            mzd = hic.getMatrixZoomData(chrom, chrom, "observed", "KR", "BP", bin_size)
            matrix = mzd.getRecordsAsMatrix(start_coord, end_coord - bin_size, start_coord, end_coord - bin_size)
            proc = process_matrix(matrix, target_width)
            summed = proc.copy() if summed is None else summed + proc
        return summed

    def load_and_sum_replicates_extended(file_list, start_coord, end_coord):
        """Load and sum Hi-C replicates for extended region (for windowed averages)"""
        summed = None
        for label, path in file_list:
            print(f"  Loading {label} (extended)...")
            hic = hicstraw.HiCFile(path)
            mzd = hic.getMatrixZoomData(chrom, chrom, "observed", "KR", "BP", bin_size)
            matrix = mzd.getRecordsAsMatrix(start_coord, end_coord - bin_size, start_coord, end_coord - bin_size)
            summed = matrix.copy() if summed is None else summed + matrix
        return summed

    # Load and process Hi-C data for visualization (original approach)
    print("Processing Hi-C data for visualization...")
    print("  KO condition:")
    ko_sum = load_and_sum_replicates(hic_files["KO"], start, end, num_bins)
    print("  WT condition:")
    wt_sum = load_and_sum_replicates(hic_files["WT"], start, end, num_bins)
    diff_matrix = ko_sum - wt_sum
    print(f"  Difference matrix shape: {diff_matrix.shape}")

    # Load and process Hi-C data for windowed averages (extended region) - only if needed
    if plot_windowed_hic_avg:
        print("Processing Hi-C data for windowed averages...")
        ext_start_coord = ext_start * bin_size
        ext_end_coord = (ext_end + 1) * bin_size
        
        print("  KO condition (extended):")
        ko_ext = load_and_sum_replicates_extended(hic_files["KO"], ext_start_coord, ext_end_coord)
        print("  WT condition (extended):")
        wt_ext = load_and_sum_replicates_extended(hic_files["WT"], ext_start_coord, ext_end_coord)
        diff_ext = ko_ext - wt_ext
        print(f"  Extended difference matrix shape: {diff_ext.shape}")

        # Calculate windowed averages
        print("Calculating windowed averages...")
        windowed_avg = np.zeros(num_bins, dtype=float)
        for i in range(num_bins):
            center = i + W  # Position in extended matrix
            lo = center - W
            hi = center + W
            if hi < diff_ext.shape[1] and lo >= 0:
                vals = diff_ext[center, lo:hi+1]
                windowed_avg[i] = np.nanmean(vals)
            else:
                windowed_avg[i] = np.nan
        
        print(f"Calculated windowed averages for {num_bins} bins")

    # CNN CONDITION MAP EXTRACTION (only if available)
    cnn_cond_map = None
    cnn_map_avg = None
    cnn_column_avg = None
    saliency_cond_map = None
    
    if has_cnn_maps:
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
        
        def extract_saliency_conditions_map(paired_maps, chromosome_num, bin_start=0, bin_end=None, row_end=None):
            """Extract saliency conditions map"""
            all_maps = paired_maps.get(chromosome_num)
            if all_maps is None:
                raise ValueError(f"No paired maps found for chromosome {chromosome_num}")
            
            # Check if saliency_conditions exists
            if "saliency_conditions" not in all_maps:
                print(f"Warning: No saliency_conditions found for chromosome {chromosome_num}")
                return None
                
            saliency_map = all_maps["saliency_conditions"]  # Shape: (224, width)
            
            if bin_end is None: 
                bin_end = saliency_map.shape[1]
            if row_end is None: 
                row_end = saliency_map.shape[0]
            half_row = saliency_map.shape[0] // 2
            # For saliency, we don't need to sum across channels (there's only 1)
            # and we don't need to take half since it's already the right shape
            return saliency_map[half_row:row_end, bin_start:bin_end]
        
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
        
        # Extract saliency conditions map if requested
        if has_saliency:
            print("Extracting saliency conditions map...")
            saliency_cond_map = extract_saliency_conditions_map(
                paired_maps=paired_maps,
                chromosome_num=chrom_num,
                bin_start=start_bin,
                bin_end=start_bin + num_bins
            )
            if saliency_cond_map is not None:
                print(f"Saliency conditions map shape: {saliency_cond_map.shape}")
            else:
                has_saliency = False
        
        # Calculate CNN condition map averages along 45-degree diagonal (only if requested)
        if plot_cnn_diagonal_avg:
            print("Calculating CNN condition map averages along diagonal...")

            H, W = cnn_cond_map.shape
            num_bins_cnn = W  # x-axis represents diagonal bins
            cnn_map_avg = np.zeros(num_bins_cnn)
            
            for x_pos in range(num_bins_cnn):
                diagonal_values = []
            
                # Sample along 45° diagonal (↗): 
                for offset in range(min(W - x_pos, H)):
                    x = x_pos + offset
                    y = offset
                    diagonal_values.append(cnn_cond_map[y, x])
            
                # Sample along 135° diagonal (↖): 
                for offset in range(min(x_pos + 1, H)):
                    x = x_pos - offset
                    y = offset
                    diagonal_values.append(cnn_cond_map[y, x])
            
                # Remove duplicate center pixel if it was sampled twice
                diagonal_values = list(set(diagonal_values))
            
                if diagonal_values:
                    cnn_map_avg[x_pos] = np.nanmean(diagonal_values)
                else:
                    cnn_map_avg[x_pos] = np.nan
            
            print(f"Computed average features from 45° and 135° diagonals for {num_bins_cnn} bins.")

        # NEW: Calculate CNN condition map column averages (only if requested)
        if plot_cnn_column_avg:
            print("Calculating CNN condition map column averages...")
            
            H, W = cnn_cond_map.shape
            cnn_column_avg = np.zeros(W)
            
            # Calculate average for each column (bin)
            for col in range(W):
                column_values = cnn_cond_map[:, col]
                # Remove NaN values before averaging
                valid_values = column_values[~np.isnan(column_values)]
                if len(valid_values) > 0:
                    cnn_column_avg[col] = np.mean(valid_values)
                else:
                    cnn_column_avg[col] = np.nan
            
            print(f"Computed column averages for {W} bins.")

    
    # FIXED: Determine actual height for Hi-C plotting
    if has_cnn_maps:
        # Align difference matrix with CNN map
        icom_diff = diff_matrix[:cnn_cond_map.shape[0], :cnn_cond_map.shape[1]]
        actual_height = cnn_cond_map.shape[0]
    else:
        # Use full Hi-C matrix height when no CNN maps
        icom_diff = diff_matrix
        actual_height = int(patch_len/2)
    
    print(f"Aligned difference matrix shape: {icom_diff.shape}")
    print(f"Actual plotting height: {actual_height}")

    # DISTANCE CALCULATIONS - Support multiple sources
    distance_dataframes = []
    
    def test_model(model, dataloader, distance_measure_local=distance_measure):
        """Test Siamese model and return distances - ONE DISTANCE PER TILE"""
        all_distances = []
        all_labels = []
        print("Running model inference...")
        
        for batch_idx, (x1, x2, y) in enumerate(dataloader):
            if batch_idx % 5 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            y = y.float()
            with torch.no_grad():
                o1, o2 = model(x1, x2)
                if distance_measure_local.lower() == "cosine":
                    dists = 1 - F.cosine_similarity(o1, o2)
                elif distance_measure_local.lower() == "pairwise":
                    dists = F.pairwise_distance(o1, o2)
                else:
                    raise ValueError(f"Unknown distance measure: {distance_measure_local}")
            
            # KEY CHANGE: No expansion - one distance per tile
            all_distances.append(dists.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        
        return np.concatenate(all_distances), np.concatenate(all_labels)

    def calculate_distances_from_model(model_path, mlhic_data, ref_genome, dist_measure, title_suffix=""):
        """Calculate distances from a single model using middle bin assignment"""
        print(f"Loading Siamese model from: {model_path}")
        model_class = getattr(models, model_name)
        model = model_class(mask=True)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        
        print(f"Loading test dataset for {title_suffix}...")
        siamese = SiameseHiCDataset(mlhic_data, reference=ref_genome)
        
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
        
        if len(siamese_subset) == 0:
            print(f"No data found for chromosome {chrom}, returning empty DataFrame")
            return pd.DataFrame()
        
        dl = DataLoader(siamese_subset, batch_size=100, sampler=SequentialSampler(siamese_subset))

        print(f"Running model testing for {title_suffix}...")
        distances, labels_siam = test_model(model, dl, distance_measure_local=dist_measure)

        # Get genomic positions
        print(f"Processing genomic positions for {title_suffix}...")
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
        
        # Get labels for the subset
        subset_labels = [siamese.labels[i] for i in subset_indices]

        # KEY CHANGE: Use middle bin positions instead of expanding
        middle_positions = [pos + (patch_len * bin_size) // 2 for pos in start_positions]
        
        # Sanity check - now we should have 1:1 correspondence
        assert len(middle_positions) == len(distances), f"Mismatch: {len(middle_positions)} positions vs {len(distances)} distances"

        # Construct DataFrame with middle bin positions
        pos_df = pd.DataFrame({
            "Chromosome": chromosomes,
            "Middle": middle_positions,  # Using middle bin positions
            "distance": distances,  # Changed from "distances" to match expected column name
            "labels": subset_labels,
            "label_siam": np.where(labels_siam==0, "within", "between")
        })
        
        print(f"Created DataFrame with {len(pos_df)} distance entries for {title_suffix} (middle bin positions)")
        return pos_df

    # Handle multiple CSV paths
    if csv_paths is not None:
        if not isinstance(csv_paths, list):
            csv_paths = [csv_paths]
        
        for i, csv_path in enumerate(csv_paths):
            print(f"Loading pre-calculated distances from: {csv_path}")
            pos_df = pd.read_csv(csv_path)
            print(f"Loaded {len(pos_df)} pre-calculated distance entries")
            
            # Add title for this dataset
            title = distance_titles[len(distance_dataframes)] if distance_titles and len(distance_dataframes) < len(distance_titles) else f"CSV Distance Plot {i+1}"
            distance_dataframes.append((pos_df, title))
    
    # Handle multiple DataFrames
    if csv_data_list is not None:
        if not isinstance(csv_data_list, list):
            csv_data_list = [csv_data_list]
            
        for i, csv_data in enumerate(csv_data_list):
            print(f"Using provided pre-calculated distances DataFrame {i+1}")
            pos_df = csv_data.copy()
            print(f"Using {len(pos_df)} pre-calculated distance entries")
            
            # Add title for this dataset
            title = distance_titles[len(distance_dataframes)] if distance_titles and len(distance_dataframes) < len(distance_titles) else f"DataFrame Distance Plot {i+1}"
            distance_dataframes.append((pos_df, title))
    
    # Handle multiple model checkpoints
    if model_ckpt_paths is not None:
        if not isinstance(model_ckpt_paths, list):
            model_ckpt_paths = [model_ckpt_paths]
        
        # Validate that we have corresponding datasets and reference genomes
        if mlhic_dataset is None or reference_genome is None:
            raise ValueError("Must provide mlhic_dataset and reference_genome when using model_ckpt_paths")
        
        # Handle distance measures
        if distance_measures is None:
            distance_measures = [distance_measure] * len(model_ckpt_paths)
        elif not isinstance(distance_measures, list):
            distance_measures = [distance_measures] * len(model_ckpt_paths)
        elif len(distance_measures) != len(model_ckpt_paths):
            raise ValueError("distance_measures must have the same length as model_ckpt_paths")
        
        # Process each model
        for i, (model_path, dist_measure) in enumerate(zip(model_ckpt_paths, distance_measures)):
            if dist_measure.lower() == "cosine":     
                dist_measure_title = "1-Cosine Similarity"
            elif dist_measure.lower() == "pairwise":
                dist_measure_title = "Pairwise Similarity"
            title = distance_titles[len(distance_dataframes)] if distance_titles and len(distance_dataframes) < len(distance_titles) else f"Model {i+1} {dist_measure_title}"
            pos_df = calculate_distances_from_model(model_path, mlhic_dataset, reference_genome, dist_measure, title)
            distance_dataframes.append((pos_df, title))
    
    # Handle legacy single model parameters (for backward compatibility)
    if (not distance_dataframes and model_ckpt_path is not None and 
        mlhic_dataset is not None and reference_genome is not None):
        
        print("Using legacy single model parameters...")
        title = distance_titles[0] if distance_titles else "Model Distance Plot"
        pos_df = calculate_distances_from_model(model_ckpt_path, mlhic_dataset, reference_genome, distance_measure, title)
        distance_dataframes.append((pos_df, title))
    
    # Validate that we have at least one distance source
    if not distance_dataframes:
        print("Warning: No distance sources provided. Only Hi-C and CNN maps will be plotted.")

    # Process all distance dataframes and calculate individual Y-axis ranges
    processed_distance_data = []
    for pos_df, title in distance_dataframes:
        # Handle both column name variants
        if 'distances' in pos_df.columns and 'distance' not in pos_df.columns:
            pos_df['distance'] = pos_df['distances']
        
        # Handle both column name variants for labels
        if 'labels_siam' in pos_df.columns and 'label_siam' not in pos_df.columns:
            pos_df['label_siam'] = pos_df['labels_siam']

        # Filter for target chromosome and range
        pos_df = pos_df[pos_df["Chromosome"] == chrom]
        print(f"Filtered to {len(pos_df)} entries for chromosome {chrom} in {title}")
        
        pos_df["bin_idx"] = ((pos_df["Middle"] / bin_size) - start_bin).astype(int)
        pos_df = pos_df[(pos_df["bin_idx"] >= 0) & (pos_df["bin_idx"] < num_bins)]
        pos_df["Mb"] = pos_df["Middle"] / 1e6
        
        print(f"Filtered to {len(pos_df)} entries within the specified range for {title}")

        # Subsample every nth bin for plotting
        print(f"Subsampling positions for plotting {title}...")
        unique_positions = sorted(pos_df["Middle"].unique())
        pos_df = pos_df[pos_df["Middle"].isin(unique_positions)].reset_index(drop=True)
        print(f"Selected {len(pos_df)} positions for plotting {title}")
        
        # Calculate individual Y-axis limits for each distance plot
        if len(pos_df) > 0:
            dist_min = pos_df["distance"].min()
            dist_max = pos_df["distance"].max()
            # Add some padding
            dist_range = dist_max - dist_min
            dist_min -= dist_range * 0.05
            dist_max += dist_range * 0.05
        else:
            dist_min, dist_max = 0, 1
        
        processed_distance_data.append((pos_df, title, dist_min, dist_max))

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
            patch_up = np.flipud(patch_up)
            if yslice.stop > pixel_size or xslice.stop > canvas_bins:
                continue
            canvas[yslice, xslice] += (patch_up if pos_or_neg == 0 else -patch_up)
        return canvas

    # Reconstruct CNN features (only if available)
    cnn_feats = None
    if has_features and has_cnn_maps:
        print("Reconstructing CNN features...")
        height, width = cnn_cond_map.shape
        chrom_num = chrom.replace("chr", "")
        
        cnn_feats = reconstruct_cnn_features(
            features=filter_features,
            paired_maps=paired_maps,
            chromosome_num=chrom_num,
            pixel_size=actual_height,
            num_bins=width
        )
        print(f"CNN features canvas shape: {cnn_feats.shape}")

    # Calculate plot dimensions
    width = num_bins
    
    # Count number of plots (conditional based on boolean parameters)
    num_plots = 1  # Always have Hi-C difference
    if plot_windowed_hic_avg:
        num_plots += 1  # Average difference line plot
    if has_cnn_maps:
        num_plots += 1  # CNN conditions map
        if plot_cnn_diagonal_avg:
            num_plots += 1  # CNN map diagonal average line plot
        if plot_cnn_column_avg:
            num_plots += 1  # CNN map column average line plot
        if has_saliency:
            num_plots += 1  # Saliency conditions map
        if has_features:
            num_plots += 2  # Features + overlay
    num_plots += len(processed_distance_data)  # Distance plots
    
    # MODIFIED: Use uniform subplot dimensions like the second function
    plot_height_per_plot = plot_w * (actual_height / chunk_size)
    fig_height = num_plots * plot_height_per_plot

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
        hic_chunk = icom_diff[:actual_height, bs:be]
        
        # Define x_bins for line plots
        x_bins = np.arange(bs, be)
        
        if has_cnn_maps:
            cnn_chunk = cnn_cond_map[:actual_height, bs:be]
        
        if has_saliency and saliency_cond_map is not None:
            saliency_chunk = saliency_cond_map[:actual_height, bs:be]
        
        # Extract feature chunk (only if available)
        feat_chunk = None
        if has_features and cnn_feats is not None:
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

        # MODIFIED: Create subplots with uniform dimensions (no custom height_ratios)
        fig, axs = plt.subplots(
            num_plots, 1,
            figsize=(plot_w, fig_height),
            constrained_layout=True
        )
        
        # Ensure axs is always a list
        if num_plots == 1:
            axs = [axs]
        
        xticks, xtick_labels = get_xlabels(bs, be, width, mb_start, mb_range)
        
        plot_idx = 0

        # 1) CNN Map (if available)
        if has_cnn_maps:
            im0 = axs[plot_idx].imshow(
                cnn_chunk, cmap='bwr', origin='lower', aspect='auto',
                vmin=-np.nanmax(np.abs(cnn_cond_map)),
                vmax=np.nanmax(np.abs(cnn_cond_map)),
                extent=[0, w, 0, actual_height]
            )
            axs[plot_idx].set_title(f"CNN Conditions Map: {mb_s:.1f}–{mb_e:.1f} Mb")
            axs[plot_idx].set_ylabel("Map Row")
            axs[plot_idx].set_ylim(0, actual_height)
            axs[plot_idx].set_xticks(xticks)
            axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
            axs[plot_idx].set_xlabel("Genomic Position (Mb)")
            plt.colorbar(im0, ax=axs[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1

            # 1.5) CNN Map Average Line Plot (Diagonal) - optional
            if plot_cnn_diagonal_avg and cnn_map_avg is not None:
                cnn_avg_chunk = cnn_map_avg[bs:be]
                
                axs[plot_idx].plot(x_bins - bs, cnn_avg_chunk, color="blue", linewidth=1.2)
                axs[plot_idx].set_title(f"CNN Conditions Avg (Diagonal): {mb_s:.1f}–{mb_e:.1f} Mb")
                axs[plot_idx].set_ylabel("Avg CNN Value")
                axs[plot_idx].set_xlim(0, w)
                axs[plot_idx].set_xticks(xticks)
                axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
                axs[plot_idx].set_xlabel("Genomic Position (Mb)")
                axs[plot_idx].grid(True, alpha=0.3)
                
                # Add horizontal line at y=0 for reference
                axs[plot_idx].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                plot_idx += 1

            # 1.6) CNN Map Column Average Line Plot - NEW
            if plot_cnn_column_avg and cnn_column_avg is not None:
                cnn_col_avg_chunk = cnn_column_avg[bs:be]
                
                axs[plot_idx].plot(x_bins - bs, cnn_col_avg_chunk, color="green", linewidth=1.2)
                axs[plot_idx].set_title(f"CNN Conditions Avg (Column): {mb_s:.1f}–{mb_e:.1f} Mb")
                axs[plot_idx].set_ylabel("Avg CNN Value")
                axs[plot_idx].set_xlim(0, w)
                axs[plot_idx].set_xticks(xticks)
                axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
                axs[plot_idx].set_xlabel("Genomic Position (Mb)")
                axs[plot_idx].grid(True, alpha=0.3)
                
                # Add horizontal line at y=0 for reference
                axs[plot_idx].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                plot_idx += 1

            # 1.75) Saliency Conditions Map (if available and requested)
            if has_saliency and saliency_cond_map is not None:
                im_sal = axs[plot_idx].imshow(
                    saliency_chunk, cmap='Reds', origin='lower', aspect='auto',
                    vmin=0,  # Saliency is typically non-negative
                    vmax=np.nanmax(saliency_cond_map),
                    extent=[0, w, 0, actual_height]
                )
                axs[plot_idx].set_title(f"Saliency Conditions Map: {mb_s:.1f}–{mb_e:.1f} Mb")
                axs[plot_idx].set_ylabel("Map Row")
                axs[plot_idx].set_ylim(0, actual_height)
                axs[plot_idx].set_xticks(xticks)
                axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
                axs[plot_idx].set_xlabel("Genomic Position (Mb)")
                plt.colorbar(im_sal, ax=axs[plot_idx], fraction=0.046, pad=0.04)
                plot_idx += 1

        # 2) Hi‑C Δ (always present)
        im1 = axs[plot_idx].imshow(
            hic_chunk, cmap='RdBu_r', origin='lower', aspect='auto',
            vmin=-np.percentile(np.abs(icom_diff), 99),
            vmax=np.percentile(np.abs(icom_diff), 99),
            extent=[0, w, 0, actual_height]
        )
        axs[plot_idx].set_title(f"Hi‑C Δ: {mb_s:.1f}–{mb_e:.1f} Mb")
        axs[plot_idx].set_ylabel("Map Row")
        axs[plot_idx].set_ylim(0, actual_height)
        axs[plot_idx].set_xticks(xticks)
        axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
        axs[plot_idx].set_xlabel("Genomic Position (Mb)")
        plt.colorbar(im1, ax=axs[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1

        # 3) Average Difference Line Plot (optional)
        if plot_windowed_hic_avg and windowed_avg is not None:
            windowed_chunk = windowed_avg[bs:be]
            
            axs[plot_idx].plot(x_bins - bs, windowed_chunk, color="black", linewidth=1.2)
            axs[plot_idx].set_title(f"Avg Δ ±{window_mb} Mb: {mb_s:.1f}–{mb_e:.1f} Mb")
            axs[plot_idx].set_ylabel(f"Avg Δ")
            axs[plot_idx].set_xlim(0, w)
            axs[plot_idx].set_xticks(xticks)
            axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
            axs[plot_idx].set_xlabel("Genomic Position (Mb)")
            axs[plot_idx].grid(True, alpha=0.3)
            
            # Add horizontal line at y=0 for reference
            axs[plot_idx].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            plot_idx += 1

        # 4) Extracted CNN Features (if available)
        if has_features and feat_chunk is not None:
            im_feat = axs[plot_idx].imshow(
                feat_chunk, cmap='bwr', origin='lower', aspect='auto',
                vmin=-np.nanmax(np.abs(cnn_feats)) if np.nanmax(np.abs(cnn_feats)) > 0 else -1,
                vmax=np.nanmax(np.abs(cnn_feats)) if np.nanmax(np.abs(cnn_feats)) > 0 else 1,
                extent=[0, w, 0, actual_height]
            )
            axs[plot_idx].set_title(f"Extracted CNN Features: {mb_s:.1f}–{mb_e:.1f} Mb")
            axs[plot_idx].set_ylabel("Map Row")
            axs[plot_idx].set_ylim(0, actual_height)
            axs[plot_idx].set_xticks(xticks)
            axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
            axs[plot_idx].set_xlabel("Genomic Position (Mb)")
            plt.colorbar(im_feat, ax=axs[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1

            # 5) Overlay (if CNN maps and features are available)
            if has_cnn_maps:
                axs[plot_idx].imshow(
                    hic_chunk, cmap='RdBu_r', origin='lower', aspect='auto',
                    vmin=-np.percentile(np.abs(icom_diff), 99),
                    vmax=np.percentile(np.abs(icom_diff), 99),
                    extent=[0, w, 0, actual_height], alpha=1.0
                )
                im2 = axs[plot_idx].imshow(
                    cnn_chunk, cmap='bwr', origin='lower', aspect='auto',
                    vmin=-np.nanmax(np.abs(cnn_cond_map)),
                    vmax=np.nanmax(np.abs(cnn_cond_map)),
                    extent=[0, w, 0, actual_height], alpha=0.6
                )
                axs[plot_idx].set_title(f"Overlay: {mb_s:.1f}–{mb_e:.1f} Mb")
                axs[plot_idx].set_ylabel("Map Row")
                axs[plot_idx].set_ylim(0, actual_height)
                axs[plot_idx].set_xticks(xticks)
                axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
                axs[plot_idx].set_xlabel("Genomic Position (Mb)")
                plt.colorbar(im2, ax=axs[plot_idx], fraction=0.046, pad=0.04)
                plot_idx += 1

        # N) Distance plots (one for each dataset) - Individual Y-axis limits
        for pos_df, title, dist_min, dist_max in processed_distance_data:
            # Filter position data for this chunk
            subdf = pos_df[(pos_df["bin_idx"] >= bs) & (pos_df["bin_idx"] < be)].copy()
            subdf["x_bin"] = subdf["bin_idx"] - bs

            if len(subdf) > 0:
                sns.lineplot(
                    data=subdf, x="x_bin", y="distance", hue="label_siam",
                    ax=axs[plot_idx]
                )
            
            axs[plot_idx].set_title(f"{title} : {mb_s:.1f}–{mb_e:.1f} Mb")
            axs[plot_idx].set_xlabel("Genomic Position (Mb)")
            axs[plot_idx].set_ylabel("Distance")
            axs[plot_idx].set_xticks(xticks)
            axs[plot_idx].set_xticklabels(xtick_labels, rotation=45)
            axs[plot_idx].set_xlim(0, w)
            axs[plot_idx].set_ylim(dist_min, dist_max)
            axs[plot_idx].legend(title="Label", fontsize=12, title_fontsize=13, loc="upper right")
            plot_idx += 1

        # Set x-axis limits for all subplots
        for ax in axs:
            ax.set_xlim(0, w)

        plt.show()
        plot_count += 1
    
    print(f"Completed plotting {plot_count} chunks!")