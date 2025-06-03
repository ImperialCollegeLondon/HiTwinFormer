import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from torch.autograd import Variable
import argparse
import json
import wandb
from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset, Augmentations
import HiSiNet.models as models
from torch_plus.loss import ContrastiveLoss
from HiSiNet.reference_dictionaries import reference_genomes


# Argument parsing
parser = argparse.ArgumentParser(description='Siamese network')
parser.add_argument('model_name', type=str, help='Model from models')
parser.add_argument('json_file', type=str, help='JSON dictionary with file paths')
parser.add_argument('learning_rate', type=float, help='Learning rate')
parser.add_argument("data_inputs", nargs='+', help="Keys for training and validation sets")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='Max training epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=0, help='Forced training epochs')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument('--seed', type=int, default=30004, help='Random seed')
parser.add_argument('--mask', type=bool, default=False, help='Mask diagonal')
parser.add_argument('--bias', type=float, default=2, help='Bias for contrastive loss')
parser.add_argument('--patience', type=int, default=3, help="How many epochs to wait after validation loss stops improving before stopping training")
parser.add_argument('--min_delta', type=float, default=1e-3,  help="Minimum improvement in validation loss required to reset patience") 
parser.add_argument('--image_size', type=int, default=224,  help="Dimensions of input tile, i.e. 224x224 (2240000 bp window size / 10kb resolution = 224), Note maxvit only works on 224x224 images") 
parser.add_argument('--loss', type=str, default="contrastive", help='Loss function')
parser.add_argument('--n_aug', type=int, default=0, help='Number of times you want to augment data with poisson/dropout')
args = parser.parse_args()

loss_function = args.loss
loss_function = loss_function.lower()

# Initialize wandb
wandb.init(
    project="HiSiNet",
    name=f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_seed{args.seed}_lf{args.loss}_num_aug{args.n_aug}",
    config=vars(args)
)

# Set device and seed
cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

# Load dataset JSON
with open(args.json_file) as json_file:
    dataset = json.load(json_file)

# Prepare datasets
train_data = [HiCDatasetDec.load(p) for name in args.data_inputs for p in dataset[name]["training"]]
val_data = [HiCDatasetDec.load(p) for name in args.data_inputs for p in dataset[name]["validation"]]
reference = reference_genomes[dataset[args.data_inputs[0]]["reference"]]

is_triplet= (loss_function=="triplet")
is_supcon= (loss_function=="supcon")
train_dataset = SiameseHiCDataset(train_data, triplet=is_triplet, reference=reference)
val_dataset = SiameseHiCDataset(val_data, triplet=is_triplet, reference=reference)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=1,  
    pin_memory=True)  
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=SequentialSampler(val_dataset), num_workers=1,  
    pin_memory=True)

# Model setup
model = eval("models." + args.model_name)(mask=args.mask, image_size=args.image_size, triplet=is_triplet, supcon=is_supcon).to(cuda)
if loss_function=="contrastive":
    nn_model = models.LastLayerNN().to(cuda)

model_path = f"{args.outpath}{args.model_name}_{args.learning_rate}_{args.batch_size}_{loss_function}_{args.seed}"
torch.save(model.state_dict(), model_path + ".ckpt")
if loss_function=="contrastive":
    torch.save(nn_model.state_dict(), model_path + "_nn.ckpt")

# Loss, optimizer, AMP, augmentor
contrastive_loss_fn = ContrastiveLoss()
classification_loss_fn = nn.CrossEntropyLoss()
triplet_loss_fn= nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
optimizer = optim.Adagrad(model.parameters())
scaler = torch.GradScaler("cuda" ,enabled=True)
augmentor = Augmentations()

# Training loop
best_val_loss = float('inf')
total_train_batches = len(train_loader)
total_val_batches = len(val_loader)
no_improved_epochs = 0

print("Started training")

for epoch in range(1, args.epoch_training + 1):
    model.train()
    if loss_function=="contrastive":
        nn_model.train()

    train_contrastive_loss = 0.0
    train_classification_loss = 0.0

    if loss_function=="contrastive":
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):

            x1_aug_list, x2_aug_list, labels_aug_list = [], [], []

            if n_aug > 0:
                for i in range(x1.shape[0]):
                    for _ in range(args.n_aug):
                        x1_aug_list.append(augmentor.normalize(augmentor(x1[i])))
                        x2_aug_list.append(augmentor.normalize(augmentor(x2[i])))
                        labels_aug_list.append(labels[i])

                x1 = torch.stack(x1_aug_list).to(cuda)
                x2 = torch.stack(x2_aug_list).to(cuda)
                labels = torch.tensor(labels_aug_list).to(cuda)
            else:
                x1 = torch.stack([augmentor.normalize(x) for x in x1]).to(cuda)
                x2 = torch.stack([augmentor.normalize(x) for x in x2]).to(cuda)
                labels = labels.to(cuda)
    
            with torch.autocast(device_type='cuda'):
                feat1, feat2 = model(x1, x2)
                logits = nn_model(feat1, feat2)
    
                class_loss = classification_loss_fn(logits, labels)
                contrastive_loss = contrastive_loss_fn(feat1, feat2, labels.float())
                total_loss = args.bias * contrastive_loss + class_loss
    
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
            train_contrastive_loss += contrastive_loss.item()
            train_classification_loss += class_loss.item()
            torch.cuda.empty_cache()
            
   
    elif loss_function == "triplet":
        for batch_idx, (x1, x2, x3) in enumerate(train_loader):
            if n_aug > 0:
                x1_aug_list, x2_aug_list, x3_aug_list = [], [], []
                for i in range(x1.shape[0]):
                    for _ in range(arg.n_aug):
                        x1_aug_list.append(augmentor.normalize(augmentor(x1[i])))
                        x2_aug_list.append(augmentor.normalize(augmentor(x2[i])))
                        x3_aug_list.append(augmentor.normalize(augmentor(x3[i])))

                x1 = torch.stack(x1_aug_list).to(cuda)
                x2 = torch.stack(x2_aug_list).to(cuda)
                x3 = torch.stack(x3_aug_list).to(cuda)
            else:
                x1 = torch.stack([augmentor.normalize(x) for x in x1]).to(cuda)
                x2 = torch.stack([augmentor.normalize(x) for x in x2]).to(cuda)
                x3 = torch.stack([augmentor.normalize(x) for x in x3]).to(cuda)

            with torch.autocast(device_type='cuda'):
                feat1, feat2, feat3 = model(x1, x2, x3)
                #logits = nn_model(feat1, feat2, feat3)
    
                #class_loss = classification_loss_fn(logits, labels)
                contrastive_loss = triplet_loss_fn(feat1, feat2, feat3)
                total_loss = contrastive_loss # args.bias * contrastive_loss + class_loss
    
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
            train_contrastive_loss += contrastive_loss.item()
            #train_classification_loss += class_loss.item()
            torch.cuda.empty_cache()

    elif loss_function=="supcon":
        # batch contains ((tile pos 1, labels pos 1),(tile pos 2, labels pos 2)...)
        for batch in train_loader:
            batch_loss = 0
            for tile in batch:
                control=[]
                treatment[]
                views, labels = tile
                for i, lab in enumerate(labels):
                    if lab==0:
                        control.append(view[i])
                    if lab==1: 
                        treatment.append(view[i])
            control_tensor   = torch.stack(control,   dim=0)   # shape [n_control,   f_dim]
            treatment_tensor = torch.stack(treatment, dim=0)   # shape [n_treatment, f_dim]
            tile_loss_info   = torch.stack([control_tensor, treatment_tensor], dim=0)
            loss_fn = MultiviewSINCERELoss()
            condition_labels = torch.tensor([0, 1], dtype=torch.long)
            tile_loss=loss_fn(tile_loss_info, condition_labels)               

    avg_train_cont = train_contrastive_loss / total_train_batches
    if loss_function=="contrastive":
        avg_train_class = train_classification_loss / total_train_batches

    # Validation
    model.eval()
    if loss_function=="contrastive":
        nn_model.eval()
    val_contrastive_loss = 0.0

    with torch.no_grad():
        if loss_function=="contrastive":
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(cuda), x2.to(cuda), labels.to(cuda).float()
                with torch.autocast(device_type='cuda'):
                    feat1, feat2 = model(x1, x2)
                    val_contrastive_loss += contrastive_loss_fn(feat1, feat2, labels).item()
                torch.cuda.empty_cache()
        elif loss_function=="triplet":
            for x1, x2, x3 in val_loader:
                x1, x2, x3 = x1.to(cuda), x2.to(cuda), x3.to(cuda)
                with torch.autocast(device_type='cuda'):
                    feat1, feat2, feat3 = model(x1, x2, x3)
                    val_contrastive_loss += triplet_loss_fn(feat1, feat2, feat3).item()
                torch.cuda.empty_cache()
            

    avg_val_cont = val_contrastive_loss / total_val_batches

    # Logging
    key = "contrastive" if loss_function=="contrastive" else "triplet"
    metrics = {
        "epoch": epoch,
        f"train_{key}_loss": avg_train_cont,
        f"val_{key}_loss":   avg_val_cont,
        "num_aug": args.n_aug
    }
    if loss_function == "contrastive":
        metrics["train_class_loss"] = avg_train_class
    wandb.log(metrics)

    #print(f"Epoch [{epoch}/{args.epoch_training}] Train Contrastive: {avg_train_cont:.6f}  "
          #f"Train Classification: {avg_train_class:.6f}  Validation Contrastive: {avg_val_cont:.6f}")

        # Save best model
    if avg_val_cont + args.min_delta < best_val_loss:
        no_improved_epochs = 0
        best_val_loss = avg_val_cont
        torch.save(model.state_dict(), model_path + ".ckpt")
        if loss_function=="contrastive":
            torch.save(nn_model.state_dict(), model_path + "_nn.ckpt")
    else:
        no_improved_epochs += 1
        
    if epoch <= args.epoch_enforced_training:
        no_improved_epochs = 0 # So patience is only started after enforced training period
        
        
    # Early stopping
    if epoch > args.epoch_enforced_training and no_improved_epochs == args.patience:
        print(f"Validation loss rose from {best_val_loss:.6f} to {avg_val_cont:.6f}. Stopping early.")
        break


wandb.finish()
