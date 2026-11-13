from src.utils import Dataset, collate_fn
from src.models import ImageEncoder, Dino, LinearProbe
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--EPOCHS", type=int, default=100)
parser.add_argument("--BATCH_SIZE", type=int, default=128)
parser.add_argument("--DROPOUT", type=float, default=0.1)
parser.add_argument("--dataset_dir", type=str, default="Dataset")
parser.add_argument("--save_dir", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--model_type", type=str)
parser.add_argument("--momentum_encoder", action='store_true')
parser.add_argument("--encoder_varient", type=str)
parser.add_argument("--task", type=str)
args = parser.parse_args()

EPOCHS=args.EPOCHS
BATCH_SIZE=args.BATCH_SIZE
DROPOUT=args.DROPOUT
TASK = args.task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

json_path = f"{args.dataset_dir}/A2_dataset/Part_Aa/Probe-Datasets/clevr_{TASK}_train.json"
img_dir = f"{args.dataset_dir}/A2_dataset/Part_Aa/Clevr_official/images/train"
img_transform = None
if args.model_type == 'clip':
    img_transform = torchvision.transforms.Resize((224, 224))
elif args.model_type == 'dino':
    img_transform = torchvision.transforms.RandomResizedCrop(96, scale=(0.4, 0.8), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
train_dataset = Dataset(json_path=json_path, img_dir=img_dir, task=TASK, img_transform=img_transform)
json_path = f"{args.dataset_dir}/A2_dataset/Part_Aa/Probe-Datasets/clevr_{TASK}_val.json"
img_dir = f"{args.dataset_dir}/A2_dataset/Part_Aa/Clevr_official/images/val"
val_dataset = Dataset(json_path=json_path, img_dir=img_dir, task=TASK, img_transform=img_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: collate_fn(batch, task=TASK, pad_idx=0, context_length=64), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, task=TASK, pad_idx=0, context_length=64), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
if args.model_type == 'clip':
    encoder = ImageEncoder(patch_size=16, num_layers=12, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=int(224*224/(16*16)), dropout=DROPOUT, encoder_varient=args.encoder_varient)
else:
    encoder = ImageEncoder(patch_size=16, num_layers=12, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=int(224*224/(16*16)), dropout=DROPOUT, encoder_varient=args.encoder_varient)
if args.model_type=='clip':
    encoder_name="ImageEncoder"
else:
    encoder_name="Teacher" if args.momentum_encoder else "Student"
encoder_state_dict = torch.load(args.model_path, map_location=device)['model']
encoder_state_dict = {
    k.replace(f"{encoder_name}.", ""): v for k, v in encoder_state_dict.items() if k.startswith(f"{encoder_name}.")
}
encoder.load_state_dict(encoder_state_dict)
model = LinearProbe(encoder=encoder, embed_dim=384, output_dim=8 if TASK=='colors' else 11, encoder_varient=args.encoder_varient).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5),
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-5, eta_min=1e-5)], milestones=[5])
loss_fn = torch.nn.CrossEntropyLoss() if TASK == 'count' else torch.nn.BCEWithLogitsLoss()

def training_epoch(model, optimizer, scheduler, loader, loss_fn, device):
    model.train()
    loss_epoch = 0
    accuracy = 0
    for batch in loader:
        X_img, X_text, X_label = batch["img"].to(device, non_blocking=True), batch["text"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad()
        y_pred = model(X_img)
        loss = loss_fn(y_pred, X_label)
        loss.backward()
        optimizer.step()
        loss_epoch+=loss.item()
        if TASK == 'count':
            accuracy+=(y_pred.argmax(dim=1)==X_label).float().mean().item()
        else:
            accuracy+=((torch.sigmoid(y_pred)>0.5).int()==X_label).float().mean().item()
    scheduler.step()
    return loss_epoch/len(loader), accuracy/len(loader)

@torch.inference_mode()
def validation_epoch(model, loader, loss_fn, device):
    model.eval()
    loss_epoch = 0;
    accuracy = 0
    for batch in loader:
        X_img, X_text, X_label = batch["img"].to(device, non_blocking=True), batch["text"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        y_pred = model(X_img)
        loss = loss_fn(y_pred, X_label)
        loss_epoch+=loss.item()
        if TASK == 'count':
            accuracy+=(y_pred.argmax(dim=1)==X_label).float().mean().item()
        else:
            accuracy+=((torch.sigmoid(y_pred)>0.5).int()==X_label).float().mean().item()
    return loss_epoch/len(loader), accuracy/len(loader)

best_validation_loss = 1e9
import time
start_time = time.time()
for epoch in range(1, EPOCHS+1):
    training_loss, training_accuracy = training_epoch(model, optimizer, scheduler, train_loader, loss_fn, device)
    validation_loss, validation_accuracy = validation_epoch(model, val_loader, loss_fn, device)
    print(f"Epoch: {epoch}| Training Loss: {training_loss}| Validation Loss: {validation_loss}: Time elapsed: {time.time()-start_time}| Training Accuracy: {training_accuracy}| Validation Accuracy: {validation_accuracy}")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy
        }, f"{args.save_dir}/epoch_{epoch:03d}.pt"
    )
    if best_validation_loss>validation_loss:
        best_validation_loss=validation_loss
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy
            }, f"{args.save_dir}/best_model.pt"
        )
end_time = time.time()
print(f"Time Taken for training: {end_time-start_time}")
