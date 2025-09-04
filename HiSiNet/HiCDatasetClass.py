import pickle
import numpy as np
from torch import as_tensor as as_torch_tensor, float as torch_float
from collections import OrderedDict
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from frozendict import frozendict
from HiSiNet.reference_dictionaries import reference_genomes
from skimage.transform import resize
from skimage.morphology import convex_hull_image
from scipy import ndimage
import hicstraw
import cooler
import torch
from collections import defaultdict
from itertools import combinations
import random
import torch.nn.functional as F

class HiCDataset(Dataset):
    """Hi-C dataset base class"""
    def __init__(self, metadata, data_res, resolution, stride=8, exclude_chroms=['chrY','chrX', 'Y', 'X', 'chrM', 'M'], reference = 'mm9', normalise = False):
        """
        Args:
        metadata: A list consisting of
            filepath: string
            replicate name: string
            norm: (one of <NONE/VC/VC_SQRT/KR>)
            type_of_bin: (one of 'BP' or 'FRAG')
            class id: containing an integer specifying the biological condition of the Hi-C file.
        data_res: The resolution for the Hi-C to be called in base pairs.
        resolution: the size of the overall region to be considered.
        stride: (optional) gives the number of images which overlap.
        """
        self.reference, self.data_res, self.resolution, self.stride_length,  self.pixel_size = reference, data_res, resolution, int(resolution/stride), int(resolution/data_res)
        self.metadata = {'filename': metadata[0], 'replicate': metadata[1], 'norm': metadata[2], 'type_of_bin': metadata[3], 'class_id': metadata[4], 'chromosomes': OrderedDict()}
        self.positions = []
        self.chrom_pos = []
        self.normalise = normalise
        self.exclude_chroms = exclude_chroms +['All', 'ALL', 'all'] 
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save(self,filename):
        with open(filename, 'wb') as output:
            output.write(pickle.dumps(self))
            output.close()
    
    def get_genomic_positions(self, append=''):
        " returns chromosome, start, end "
        try:
            chromosome_metadata = self.chromosomes.items()
        except:
            chromosome_metadata = self.metadata['chromosomes'].items()
        chromosomes = np.concatenate( [ np.repeat(append+chromname, chromosome_range[1]-chromosome_range[0]) for 
                                    chromname, chromosome_range in chromosome_metadata])
        return {"Chromosome": chromosomes, "Start":  np.array(self.positions),"End": np.array(self.positions)+self.resolution}

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            unpickled = pickle.Unpickler(file)
            loadobj = unpickled.load()
        return loadobj


class HiCDatasetDec(HiCDataset):
    """Hi-C dataset loader using hicstraw.HiCFile interface. Creates image tiles for each chromosome. loop through stride length and then extract tiles and quality control"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # open .hic file
        hic = hicstraw.HiCFile(self.metadata['filename'])
        # build chromosome size map
        chrom_sizes = {
            c.name: c.length
            for c in hic.getChromosomes()
            if c.name != "All"
        }

        # decide which chromosomes to load (use chrom names from chrom_sizes). Remove chromosomes to exclude.
        to_load = [ch for ch in chrom_sizes if ch not in self.exclude_chroms]

        # load each chromosome
        for chrom in to_load:
            self.get_chromosome_tiles(hic, chrom, chrom_sizes[chrom])

        # freeze data structures
        self.data = tuple(self.data)
        self.positions = tuple(self.positions)
        self.chrom_pos = tuple(self.chrom_pos)
        self.metadata = frozendict(self.metadata)

    def add_chromosome(self, chromosome):
        # avoid re-loading same chrom
        if (chromosome in self.metadata['chromosomes'].keys()) |  (chromosome[3:] in self.metadata['chromosomes'].keys()): return print('chromosome already loaded')
        self.data, self.positions = list(self.data), list(self.positions)
        hic = hicstraw.HiCFile(self.metadata['filename']) # Load file
        size = next(c.length for c in hic.getChromosomes() if c.name == chromosome) # getchromosome makes dictionary of chrom:length, extract first length
        self.get_chromosome_tiles(hic, chromosome, size)
        self.data = tuple(self.data)
        self.positions = tuple(self.positions)

    def get_chromosome_tiles(self, hic, chrom, chrom_length):
        # record start index for this chrom
        start_index = len(self.positions)
        # create chromosome level matrix-zoom object
        mzd = hic.getMatrixZoomData(chrom, chrom,
                                    self.metadata.get('type_of_data', 'observed'),
                                    self.metadata['norm'],
                                    'BP',
                                    self.data_res)
        # slide window across stride length to create multiple zoomed hi-c tiles
        for start in range(0, chrom_length, self.stride_length):
            end = min(start + self.resolution - self.data_res, chrom_length) # The end has -self.data_res because otherwise getRecordsAsMatrix would create a matrix of pixel_size+1
            if end - start < self.resolution - self.data_res:  # this is a truncated window at the tail — skip it
                continue
            mat = mzd.getRecordsAsMatrix(start, end, start, end)
            img = self._process_matrix(mat) # Here we process the Hi-C matrix
            image_exists = (img is not None)
            self.chrom_pos.append((chrom, chrom_length, start, image_exists))
            if img is not None:
                self.data.append((img, self.metadata['class_id'])) # store image and related class
                self.positions.append(start) # store start position


        # store indices for this chromosome
        key = chrom[3:] if chrom.startswith("chr") else chrom
        self.metadata['chromosomes'][key] = (start_index, len(self.positions))

    def _process_matrix(self, matrix):
        # mask sparse or NaN-heavy windows
        flat = matrix.flatten()
        if len(np.unique(np.where(matrix > 0)[0])) < 0.9 * self.pixel_size: # remove if more than 10% of the rows are missing any contacts
            return None
        if np.isnan(flat).sum() > 0.5 * flat.size: # Remove tiles that are more than 50% NaN
            return None

        # if we want to augment the data we don't want to normalise (make between 0-1) it yet
        mat = np.nan_to_num(matrix)

        # to tensor shape [1,C,H,W]
        tensor = torch.tensor(mat, dtype=torch.float32).unsqueeze(0)
        return tensor

class GroupedHiCDataset(HiCDataset):
    """Grouping multiple Hi-C datasets together"""
    def __init__(self, list_of_HiCDatasets):
        #self.reference = reference
        self.data,  self.metadata, self.starts, self.files = tuple(), [], [], set()
        if not isinstance(list_of_HiCDatasets, list): print("list of HiCDataset is not list type") #stop running
        self.resolution, self.data_res = list_of_HiCDatasets[0].resolution, list_of_HiCDatasets[0].data_res
        for dataset in list_of_HiCDatasets: self.add_data(dataset)

    def add_data(self, dataset):
        if not isinstance(dataset, HiCDataset): return print("file not HiCDataset")
        #if self.reference != dataset.reference: return print("incorrect reference")
        if self.resolution != dataset.resolution: return print("incorrect resolution")
        if self.data_res != dataset.data_res: return print("data resolutions do not match")
        self.data = self.data + dataset.data # Combine the dataset's tiles once all checks have been made
        self.metadata.append(dataset.metadata) # Add metadata
        self.starts.append(len(self.data)) # Add start index

class SiameseHiCDataset(HiCDataset):
    """Dataset creation class to make either pairs, triplets, or all tile locations for one location for SupCon"""
    def __init__(self, list_of_HiCDatasets, triplet=False, supcon=False, sims=(0,1), reference = reference_genomes["mm9"]):
        self.triplet = triplet #if using triplet loss data should be created differently
        self.supcon = supcon
        self.sims = sims
        self.reference, self.chromsizes = reference
        self.data =[]
        self.positions =[]
        self.pos = []
        self.labels = []
        self.chromosomes = OrderedDict()
        checks = self.check_input(list_of_HiCDatasets)
        if not checks: return None
        self.resolution, self.data_res, self.stride_length = list_of_HiCDatasets[0].resolution, list_of_HiCDatasets[0].data_res, list_of_HiCDatasets[0].stride_length
        self.make_data(list_of_HiCDatasets) # Here is where the siamese dataset is made
        self.metadata = tuple([data.metadata for data in list_of_HiCDatasets])

    def check_input(self, list_of_HiCDatasets):
        filenames_norm = set()
        if not isinstance(list_of_HiCDatasets, list): print("list of HiCdatasets is not list type")
        for data in list_of_HiCDatasets:
            if not isinstance(data, HiCDataset):
                print("List of HiCDatasets need to be a list containing only HiCDataset objects.")
                return False
            # if (data.metadata['filename'], data.metadata['norm']) in filenames_norm:
            #     print("file has been passed twice with the same normalisation") #maybe make this a warning instead of not doing it
            filenames_norm.add((data.metadata['filename'], data.metadata['norm']))
        return True

    def check_input_parameters(self, dataset): #where we check if the dataset is compatible with what we want to do
        pass


    def append_data(self, curr_data, pos, chrom):
        """
        curr_data: list of (image, class_label) for a single genomic coordinate
        pos: genomic coordinate
        """
            
        if self.triplet:
            # Group samples by their class label
            by_label = defaultdict(list)
            for idx, (img, lbl) in enumerate(curr_data):
                by_label[lbl].append((idx, img))
            # For each class that has at least 2 samples, form anchor–positive pairs,
            # then for each such pair, pair with all negatives from other classes.
            for lbl, samples in by_label.items():
                # If fewer than 2 replicates, no anchor–positive possible
                if len(samples) < 2:
                    continue
                # all combinations of two within this class
                for (i, img_i), (j, img_j) in combinations(samples, 2):
                    # negatives = samples in *other* classes
                    for other_lbl, other_samples in by_label.items():
                        if other_lbl == lbl:
                            continue # ensure negative is not from the same label
                        for k, img_k in other_samples:
                            # two triplets: (i anchor, j positive, k negative) and vice versa
                            self.data.append((img_i, img_j, img_k))
                            self.positions.append(pos)
                            self.labels.append((i, j, k))

                            self.data.append((img_j, img_i, img_k))
                            self.positions.append(pos)
                            self.labels.append((j, i, k))

        elif self.supcon:


            # 1) bucket images by label
            by_label = defaultdict(list)
            for img, lbl in curr_data:
                by_label[lbl].append(img)
        
            # 2) require at least 2 of each
            if len(by_label[0]) < 2 or len(by_label[1]) < 2:
                return  # skip this tile
        
            # 3) flatten both classes in one go
            #    tiles  = [ all imgs of cls0, then all imgs of cls1 ]
            #    labs   = [ 0 repeated len(cls0) , 1 repeated len(cls1) ]
            tiles, labs = zip(*(
                (img, lbl)
                for lbl in (0, 1)
                for img in by_label[lbl]
            ))
        
            # 4) stack into tensors
            tile_tensor  = torch.stack(tiles, dim=0)            # [n_views, C, H, W]
            label_tensor = torch.tensor(labs, dtype=torch.long) # [n_views]
        
            # 5) append just like the other branches
            self.data.append((tile_tensor, label_tensor))
            self.positions.append(pos)
            self.pos.append((pos, chrom))
                            
      
        else: # For regular contrastive loss
            self.data.extend([(curr_data[k][0], curr_data[j][0], (self.sims[0] if curr_data[k][1] == curr_data[j][1] else self.sims[1]) ) for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))]) # Loop through pairs of data
            #For explainability: 
            #for k in range(len(curr_data)): 
            #    for j in range(k + 1, len(curr_data)): make non-redundant pairs of data indices k, j
            #        img_k = curr_data[k][0]  # image from dataset k
            #        img_j = curr_data[j][0]  # image from dataset j
            #        label = self.sims[0] if curr_data[k][1] == curr_data[j][1] else self.sims[1]  # similar or dissimilar
            #        self.data.append((img_k, img_j, label))

            self.positions.extend( [pos for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))]) # add position # list to [positions] for number of pairs of data created
            self.pos.extend([(pos, chrom) for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])
            self.labels.extend( [( k, j) for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))]) # add labels list to [labels] of which pairs are being compared 

    def make_data(self, list_of_HiCDatasets):
        datasets = len(list_of_HiCDatasets)
        
        for chrom in self.chromsizes.keys(): # For each chromosome
            start_index = len(self.positions) # starts at 0 
            starts, positions = [], [] # Start indexes and positions (say index 1, 1000bp)
            for i in range(0, datasets): # For each dataset
                start, end = list_of_HiCDatasets[i].metadata['chromosomes'].setdefault(chrom, (0,0)) # Fetch the start and end indices for the chromosome
                starts.append(start)
                positions.append(list(list_of_HiCDatasets[i].positions[start:end])) # Extract all start positions of all index position tiles in a dataset (from start to end index)
            for pos in range(0, self.chromsizes[chrom], self.stride_length)[::-1]: # For each position in each chromosome in reverse order
                curr_data = []
                for i in range(0,datasets): # For each dataset
                    if positions[i][-1:]!=[pos]: continue # If the last start position in [positions] doesn't match pos in the last position, then skip it.
                    curr_data.append(list_of_HiCDatasets[i][starts[i]+len(positions[i])-1] ) # Grab the relevant tile and class ID and add to current data 
                    positions[i].pop() # Remove last index position from positions list and continue loop to match positions[-1:] to pos
                self.append_data(curr_data, pos, chrom) # Now, once all instances of the same start position is found from each dataset being compared, create pairs of data at that position
            self.chromosomes[chrom] = (start_index,len(self.positions)) # start index to end index (self.positions has new number of positions after append_data)
        self.data = tuple(self.data) # Fixed data now

class HiCDatasetCool(HiCDataset):
    """Hi-C dataset loader"""
    def __init__(self, metadata, resolution, **kwargs):
        """ metadata: A list consisting of
            filepath: string
            replicate name: string
            norm: (one of <None/cool_norm/KR/VC/VC_SQRT>)
            class id: containing an integer specifying the biological condition of the Hi-C file."""
        cl_file= cooler.Cooler(metadata[0])
        metadata[2] = True if (metadata[2]=="cool_norm") else metadata[2] 
        metadata.insert(3, "NA")
        super(HiCDatasetCool, self).__init__(metadata, cl_file.binsize, resolution, reference = cl_file.info["genome-assembly"], **kwargs)
        chromosomes = list(set(cl_file.chromnames) - set(self.exclude_chroms))
        for chromosome in chromosomes: self.get_chromosome(cl_file, chromosome)
        self.data, self.metadata, self.positions = tuple(self.data), frozendict(self.metadata), tuple(self.positions)

    def add_chromosome(self, chromosome):
        if (chromosome in self.metadata['chromosomes'].keys()) |  (chromosome[3:] in self.metadata['chromosomes'].keys()): return print('chromosome already loaded')
        self.data, self.positions = list(self.data), list(self.positions)
        cl_file = cooler.Cooler(self.metadata['filename'])
        self.get_chromosome(cl_file, chromosome)
        self.data, self.positions = tuple(self.data), tuple(self.positions)

    def get_chromosome(self, cl_file, chromosome):
        stride = int(self.stride_length/self.data_res)
        cl_matrix = cl_file.matrix(balance = self.metadata['norm'])
        first, last = cl_file.extent(chromosome)
        initial = len(self.positions)
        self.metadata['chromosomes'][chromosome] = []
        for start_pos in range(first, last, stride): self.make_matrix(cl_matrix,  start_pos, first)
        self.metadata['chromosomes'][chromosome]= (initial, len(self.positions))

    def make_matrix(self, cl_matrix, start_pos, first):
        image_scp = cl_matrix[start_pos:start_pos+self.pixel_size, start_pos:start_pos+self.pixel_size]
        if (sum(np.diagonal(np.isnan(image_scp)|(image_scp==0))) > self.pixel_size*0.9) : return None
        image_scp[np.isnan(image_scp)] = 0
        image_scp = image_scp/np.nanmax(image_scp)
        image_scp = np.expand_dims(image_scp, axis=0)
        image_scp = as_torch_tensor(image_scp, dtype=torch_float)
        self.data.append((image_scp, self.metadata['class_id']))
        self.positions.append( int(self.data_res*(start_pos-first)))

class PairOfDatasets(SiameseHiCDataset):
    def __init__(
        self,
        list_of_HiCDatasets,
        model,
        diagonal_off=3,
        compute_saliency=False,
        distance_measure="pairwise",  # <-- NEW
        device=None,
        **kwargs
    ):
        """
        Feature and saliency map creation
        Args:
            list_of_HiCDatasets: list of HiCDataset instances
            model: trained Siamese model
            diagonal_off: int, diagonal mask offset
            compute_saliency: bool, if False skips IG computations
            distance_measure: "pairwise" (Euclidean) or "cosine"
            device: torch.device or None (auto-detect)
        """
        print("[PairOfDatasets] 1: entering __init__", flush=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self._diag_off = diagonal_off
        self.compute_sal = compute_saliency
        self.distance_measure = distance_measure.lower()
        self.saliency = []  # filled in append_data

        print(f"[PairOfDatasets] 2: model set on {self.device}", flush=True)

        reference = kwargs.pop("reference", reference_genomes["mm9"])
        super().__init__(list_of_HiCDatasets, reference=reference)
        print(f"[PairOfDatasets] 3: feature/saliency calculations complete", flush=True)
        self.pixel_size = int(self.resolution / self.data_res)
        print(f"[PairOfDatasets] 4: Start map creation", flush=True)
        self.paired_maps = {
            chrom: self.make_maps_combined(chrom)
            for chrom in self.chromosomes
        }
        print("[PairOfDatasets] init complete", flush=True)

    def _compute_distance(self, o1, o2):
        """Helper: compute distance between embeddings for use in integrated gradients calculation"""
        if self.distance_measure == "cosine":
            return 1 - F.cosine_similarity(o1, o2)
        elif self.distance_measure == "pairwise":
            return F.pairwise_distance(o1, o2)
        else:
            raise ValueError(f"Unknown distance measure: {self.distance_measure}")

    def integrated_gradients(self, x1, x2, n_steps=50):
        """Compute 2D IG saliency for embedding‐distance between x1,x2."""
        #x1, x2 = x1.to(self.device), x2.to(self.device)
        tg1 = torch.zeros_like(x1, device=self.device)
        tg2 = torch.zeros_like(x2, device=self.device)

        for alpha in np.linspace(0, 1, n_steps):
            xi1 = (x2 + alpha * (x1 - x2)).detach().requires_grad_(True)
            xi2 = (x1 + alpha * (x2 - x1)).detach().requires_grad_(True)

            # embeddings
            f1_xi1 = self.model.forward_one(xi1)
            f1_x2  = self.model.forward_one(x2)
            f1_x1  = self.model.forward_one(x1)
            f1_xi2 = self.model.forward_one(xi2)

            # distance based on user choice
            d1 = self._compute_distance(f1_xi1, f1_x2).sum()
            d2 = self._compute_distance(f1_x1,  f1_xi2).sum()

            self.model.zero_grad(); d1.backward(retain_graph=True)
            tg1 += xi1.grad; xi1.grad.zero_()
            self.model.zero_grad(); d2.backward()
            tg2 += xi2.grad; xi2.grad.zero_()

        ag1 = tg1 / n_steps
        ag2 = tg2 / n_steps
        ig1 = (x1 - x2) * ag1
        ig2 = (x2 - x1) * ag2
        comb = 0.5 * (ig1 + ig2)
        heatmap = comb.mean(dim=1).squeeze()
        return heatmap

    def append_data(self, curr_data, pos, chrom):
        feature_list, saliency_list = [], []
    
        if len(curr_data) < 2:
            return
    
        H = curr_data[0][0].shape[-1]
        band = (
            np.tril(np.ones((H, H)), k=-self._diag_off)
            + np.triu(np.ones((H, H)), k=self._diag_off)
        ).astype(float)
        diag_mask = torch.from_numpy(band).float().to(self.device)
    
        for i, (img_i, ci) in enumerate(curr_data):
            for j, (img_j, cj) in enumerate(curr_data[i + 1:], start=i + 1):
                
                mi = img_i.to(self.device)
                mj = img_j.to(self.device)
    
                if mi.max() > 0:
                    mi = mi / mi.max()
                if mj.max() > 0:
                    mj = mj / mj.max()
    
                # ---- Apply diagonal mask ----
                mi = mi * diag_mask
                mj = mj * diag_mask
    
                # ---- Feature maps (if available) ----
                if hasattr(self.model, "features"):
                    fi = self.model.features(mi.unsqueeze(0)).abs()
                    fj = self.model.features(mj.unsqueeze(0)).abs()
    
                    if ci != cj:
                        diff = fi - fj if (ci == 1 and cj == 0) else fj - fi
                    else:
                        diff = fi - fj
    
                    feature_list.append(diff)  # keep on GPU
                else:
                    feature_list.append(torch.zeros((1, H, H), device=self.device))
    
                # ---- Saliency maps (if available) ----
                if self.compute_sal and ci != cj:
                    s_map = self.integrated_gradients(
                        mi.unsqueeze(0), mj.unsqueeze(0), n_steps=100
                    )
                    saliency_list.append(s_map)  # keep on GPU
                else:
                    #saliency_list.append(torch.zeros((H, H), device=self.device))
                    pass
    
        # Move everything to CPU / numpy **once** after loop
        feature_cpu = [f.detach().cpu().numpy()[0] for f in feature_list]
        saliency_cpu = [s.detach().cpu().numpy() for s in saliency_list]
    
        L = len(feature_cpu)
        self.data.extend(feature_cpu)
        self.saliency.extend(saliency_cpu)
        self.positions.extend([pos] * L)
        self.labels.extend([
            (a, b) for a in range(len(curr_data)) for b in range(a + 1, len(curr_data))
        ])
        self.pos.extend([(pos, chrom)] * L)

    def make_maps_combined(self, chromosome):
        print(f"Started creating differential feature map for {chromosome}")
        nfilter = self.model.features[-2].out_channels
        chrom_i, chrom_j = self.chromosomes[chromosome]
        if chrom_i == chrom_j:
            return None
    
        length    = int((self.positions[chrom_i] + self.resolution) / self.data_res)
        dims_feat = (nfilter, self.pixel_size, length)
        dims_sal  = (self.pixel_size, length)
    
        # allocate both accumulators and norms
        grouped = {
            "replicate"          : np.zeros(dims_feat),
            "conditions"         : np.zeros(dims_feat),
            #"saliency_replicate" : np.zeros(dims_sal),
            "saliency_conditions": np.zeros(dims_sal),
        }
        norms = {
            "replicate"          : np.zeros(dims_feat),
            "conditions"         : np.zeros(dims_feat),
            #"saliency_replicate" : np.zeros(dims_sal),
            "saliency_conditions": np.zeros(dims_sal),
        }
    
        for idx in range(chrom_j, chrom_i, -1):
            true    = idx - 1
            pair    = self.labels[true]
            pos_bin = int(self.positions[true] / self.data_res)
    
            typ = ("replicate"
                   if self.metadata[pair[0]]["class_id"]
                      == self.metadata[pair[1]]["class_id"]
                   else "conditions")
    
            # feature maps
            for f in range(nfilter):
                x = self.data[true][f]
                x = (x + x.T) / 2
                rot_feat = ndimage.rotate(x, 45, reshape=True)
                small_f  = resize(rot_feat,
                                  (self.pixel_size, self.pixel_size),
                                  preserve_range=True,
                                  anti_aliasing=False)
                grouped[typ][f, :, pos_bin:pos_bin+self.pixel_size] += small_f
                norms[typ][f, :, pos_bin:pos_bin+self.pixel_size]   += 1
    
            # saliency maps
            if self.compute_sal:
                s = self.saliency[true]
                H, W = s.shape
                band = (
                    np.tril(np.ones((H, W)), k=-self._diag_off)
                    + np.triu(np.ones((H, W)), k=self._diag_off)
                )
                mask2d = band.astype(bool)
                s0 = np.where(mask2d, s, 0.0)
                rot_s = ndimage.rotate(s0, 45, reshape=True)
                small_s = resize(rot_s,
                                 (self.pixel_size, self.pixel_size),
                                 preserve_range=True,
                                 anti_aliasing=False)
                sal_key = f"saliency_{typ}"
                grouped[sal_key][:, pos_bin:pos_bin+self.pixel_size] += small_s
                norms[sal_key][:, pos_bin:pos_bin+self.pixel_size]   += 1
    
        # normalize element-wise
        for k in grouped:
            with np.errstate(divide='ignore', invalid='ignore'):
                grouped[k] = np.where(norms[k] > 0, grouped[k] / norms[k], 0.0)
    
        print(f"created grouped map for {chromosome}", flush=True)
        return grouped

    def extract_features(self, chromosome, nfilter, pair, qthresh=0.999, min_length=10, max_length=256, min_width=10,max_width=256, pad_extra=3, im_size=20):
        if nfilter=="all": curr_map = np.concatenate([np.sum(value[pair][:,:int(self.pixel_size/2),:],axis=0) for chrom, value in self.paired_maps.items() 
                            if value is not None ], axis=1) # create a global paired map for thresholding
        else: curr_map = np.concatenate([value[pair][nfilter,:int(self.pixel_size/2),:] for chrom, value in self.paired_maps.items() 
                            if value is not None ], axis=1)
        band = np.ones_like(curr_map, dtype=bool)
        band[-pad_extra:, :] = False  # Mask out the bottom 3 rows
        curr_map = np.where(band, curr_map, np.nan)
        pos_thresh = np.max([np.nanquantile(curr_map, qthresh),-np.nanquantile(curr_map, 1-qthresh)])
        neg_thresh = np.min([-np.nanquantile(curr_map, qthresh),np.nanquantile(curr_map, 1-qthresh)]) # calculate global intensity threshold

        if nfilter=="all": curr_map=np.sum(self.paired_maps[chromosome][pair][:,int(self.pixel_size/2):,:],axis=0)
        else: curr_map=self.paired_maps[chromosome][pair][nfilter,int(self.pixel_size/2):,:]
        band = np.ones_like(curr_map, dtype=bool)
        band[-pad_extra:, :] = False
        curr_map = np.where(band, curr_map, np.nan)
        arr_pos, arr_neg =  (curr_map>pos_thresh), (curr_map<neg_thresh) # Filter by threshold
        arr_pos, arr_neg =  ndimage.label(arr_pos), ndimage.label(arr_neg) # Connected component analysis to get individual features

        features = []
        for pos_or_neg, arr in enumerate([arr_pos, arr_neg]):
            for feature_index in np.unique(arr[0])[1:]:
                indices=np.where(arr[0]==feature_index)
                if ((max(indices[0])-min(indices[0]))<=min_length )|((max(indices[1])-min(indices[1]))<=min_width): continue 
                if ((max(indices[0])-min(indices[0]))>=max_length )|((max(indices[1])-min(indices[1]))>=max_width): continue  
                temp = convex_hull_image(arr[0]==feature_index) # convex hulling
                temp = temp[np.min(indices[0]):np.max(indices[0]), np.min(indices[1]):np.max(indices[1])]

                if (temp.shape[0]<=min_length )|(temp.shape[1]<=min_width): continue
                if (temp.shape[0]>=max_length )|(temp.shape[1]>=max_width): continue

                original_dims, height= temp.shape, np.min(indices[0])
                temp = resize(temp, (im_size,im_size),anti_aliasing=False,preserve_range=True)
                temp = np.flipud(temp) # create resized figures
                features.append((
                    feature_index,              # ID
                    temp,                       # Resized convex hull mask
                    original_dims,              # Original shape before resizing
                    height,                     # Vertical position
                    arr[1],                     # Number of features (label set)
                    pos_or_neg,                 # 0 = positive, 1 = negative
                    qthresh,                    # Quantile intensity threshold used
                    [chromosome,                # Genomic location metadata, chromosome number
                     np.min(indices[0]),        # y0
                     np.max(indices[0]),        # y1
                     np.min(indices[1]),        # x0    
                     np.max(indices[1])]        # x1
                ))
        return features
        
class Augmentations:
    """Data augmentation class for ML training. Only realistic augmentations are applied, which include poisson and random dropout. Takes in 3D tensor (1, H, W)"""
    def __init__(self):
        pass

    def __call__(self, matrix, force_augmentation=False):
        if force_augmentation == True:
            rand = random.randrange(2)
            if rand == 0:
                augmented = self.poisson_resample(matrix)
            else:
                augmented = self.random_dropout(matrix)
    
        else:  
            rand = random.randrange(3)
            if rand == 0:
                augmented = matrix
            elif rand == 1:
                augmented = self.poisson_resample(matrix)
            else:
                augmented = self.random_dropout(matrix)

        return augmented

    @staticmethod
    def poisson_resample(matrix, normalize_total=True):
        if torch.is_tensor(matrix):
            matrix_2d = matrix.squeeze(0).clone()
            lower_triangle = torch.tril(matrix_2d)
            aug = torch.poisson(lower_triangle)
            diag = torch.diag(torch.diag(aug))
            transformed = aug + aug.T - diag
            if normalize_total:
                orig_sum = matrix_2d.sum()
                new_sum = transformed.sum()
                if new_sum > 0:
                    transformed *= (orig_sum / new_sum)
                    
        return transformed.unsqueeze(0)

    @staticmethod
    def random_dropout(matrix, dropout_lower_bound=0.025, dropout_upper_bound=0.075):
        
        dropout = random.uniform(dropout_lower_bound, dropout_upper_bound)

        if torch.is_tensor(matrix):
            matrix_2d = matrix.squeeze(0).clone()
            non_zero = torch.nonzero(matrix_2d, as_tuple=False)
            n = int(dropout * non_zero.size(0))
            if n > 0:
                chosen = non_zero[torch.randperm(non_zero.size(0))[:n]]
                matrix_2d[chosen[:, 0], chosen[:, 1]] = 0

        return matrix_2d.unsqueeze(0)
    
    @staticmethod
    def normalize(matrix):
        """Normalize batch of tensors (B,C,H,W) to [0, 1] by dividing by max."""
        if torch.is_tensor(matrix):
            max_vals = matrix.amax(dim=(1,2,3), keepdim=True)
            return matrix / max_vals 
        else:
            raise TypeError("Input must be a PyTorch tensor")