from src.utils import Dataset, collate_fn
from src.models import ImageEncoder, Dino
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--EPOCHS", type=int, default=100)
parser.add_argument("--BATCH_SIZE", type=int, default=128)
parser.add_argument("--DROPOUT", type=float, default=0.1)
parser.add_argument("--dataset_dir", type=str, default="Dataset")
parser.add_argument("--save_dir", type=str)
parser.add_argument("--encoder_varient", type=str)
args = parser.parse_args()

EPOCHS=args.EPOCHS
BATCH_SIZE=args.BATCH_SIZE
DROPOUT=args.DROPOUT

print(f"EPOCHS:{EPOCHS}| BATCH_SIZE: {BATCH_SIZE}| DROPOUT: {DROPOUT}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

json_path = f"{args.dataset_dir}/A2_dataset/Part_A/train/clevr_train_captions.json"
img_dir = f"{args.dataset_dir}/A2_dataset/Part_A/train/images"
train_dataset = Dataset(json_path=json_path, img_dir=img_dir, task="DinoPretraining", img_transform=None)
json_path = f"{args.dataset_dir}/A2_dataset/Part_A/val/clevr_val_captions.json"
img_dir = f"{args.dataset_dir}/A2_dataset/Part_A/val/images"
val_dataset = Dataset(json_path=json_path, img_dir=img_dir, task="DinoPretraining", img_transform=None)
json_path = f"{args.dataset_dir}/A2_dataset/Part_A/train/CLEVR_scenes.json"
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: collate_fn(batch, task="DinoPretraining", pad_idx=0, context_length=64), num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, task="DinoPretraining", pad_idx=0, context_length=64), num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)

model = Dino(
    ImageEncoder(patch_size=16, num_layers=12, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=int(224*224/(16*16)), dropout=DROPOUT, encoder_varient=args.encoder_varient),
    ImageEncoder(patch_size=16, num_layers=12, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=int(224*224/(16*16)), dropout=DROPOUT, encoder_varient=args.encoder_varient),
    img_dim=384,
    embed_dim=4096,
    teacher_temperature=0.06,
    student_temperature=0.1,
    center_momentum=0.95,
    encoder_varient=args.encoder_varient
).to(device)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    import math
    def lr_update(current_step):
        if current_step < num_warmup_steps:
            return float(current_step)/float(max(1, num_warmup_steps))
        return 0.5*(1.0+math.cos(math.pi*float(current_step-num_warmup_steps)/float(max(1, num_training_steps-num_warmup_steps))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)

optimizer = torch.optim.AdamW(list(set(model.parameters())-(set(model.Teacher.parameters()).union(model.teacher_projection.parameters()))), lr=5e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
schedular = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*EPOCHS*len(train_loader)), num_training_steps=EPOCHS*len(train_loader))

momentum_array=np.array([1+0.5*(0.996-1)*(1+np.cos(np.pi*i/(EPOCHS*len(train_loader)))) for i in range(EPOCHS*len(train_loader))])
it=0

def training_epoch(model, optimizer, schedular, loader, device):
    global it
    model.train()
    loss_epoch = 0;
    for batch in loader:
        X_img, X_text, X_label = tuple(img.to(device, non_blocking=True) for img in batch["img"]), batch["text"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = model(X_img)
        loss.backward()
        optimizer.step()
        schedular.step()
        with torch.no_grad():
            momentum=momentum_array[it] if it < len(momentum_array) else 1
            it+=1
            for student_params, teacher_params in zip(model.Student.parameters(), model.Teacher.parameters()):
                teacher_params.data = teacher_params.data*momentum+student_params.data*(1-momentum)
            for student_params, teacher_params in zip(model.student_projection.parameters(), model.teacher_projection.parameters()):
                teacher_params.data = teacher_params.data*momentum+student_params.data*(1-momentum)
        loss_epoch+=loss.item()
    return loss_epoch/len(loader)

@torch.inference_mode()
def validation_epoch(model, loader, device):
    model.eval()
    loss_epoch = 0;
    for batch in loader:
        X_img, X_text, X_label = tuple(img.to(device, non_blocking=True) for img in batch["img"]), batch["text"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        loss = model(X_img)
        loss_epoch+=loss.item()
    return loss_epoch/len(loader)

best_validation_loss = 1e9
import time
start_time = time.time()
for epoch in range(1, EPOCHS+1):
    training_loss = training_epoch(model, optimizer, schedular, train_loader, device)
    validation_loss = validation_epoch(model, val_loader, device)
    print(f"Epoch: {epoch}| Training Loss: {training_loss}| Validation Loss: {validation_loss}: Time elapsed: {time.time()-start_time}")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedular": schedular.state_dict(),
        "training_loss": training_loss,
        "validation_loss": validation_loss
        }, f"{args.save_dir}/epoch_{epoch:03d}.pt"
    )
    if best_validation_loss>validation_loss:
        best_validation_loss=validation_loss
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "training_loss": training_loss,
            "validation_loss": validation_loss
            }, f"{args.save_dir}/best_model.pt"
        )
end_time = time.time()
print(f"Time Taken for training: {end_time-start_time}")
