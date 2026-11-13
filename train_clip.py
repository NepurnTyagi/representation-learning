from src.utils import Dataset, collate_fn, BatchSampler
from src.models import TextEncoder, ImageEncoder, Clip
import torch
from torch.utils.data import DataLoader
import torchvision
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
img_transform = torchvision.transforms.Resize((224, 224))
train_dataset = Dataset(json_path=json_path, img_dir=img_dir, task="ClipPretraining", img_transform=img_transform, tokenizer_path="tokenizer.json")
json_path = f"{args.dataset_dir}/A2_dataset/Part_A/val/clevr_val_captions.json"
img_dir = f"{args.dataset_dir}/A2_dataset/Part_A/val/images"
val_dataset = Dataset(json_path=json_path, img_dir=img_dir, task="ClipPretraining", img_transform=img_transform, tokenizer_path="tokenizer.json")
json_path = f"{args.dataset_dir}/A2_dataset/Part_A/train/CLEVR_scenes.json"
batch_sampler = BatchSampler(train_dataset, json_path = json_path, batch_size = BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, collate_fn=lambda batch: collate_fn(batch, task="ClipPretraining", pad_idx=0, context_length=64))
batch_sampler = BatchSampler(val_dataset, json_path = json_path, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_sampler=batch_sampler, collate_fn=lambda batch: collate_fn(batch, task="ClipPretraining", pad_idx=0, context_length=64))
vocabulary = train_dataset.getVocabulary()

model = Clip(
    TextEncoder(vocab_size=len(vocabulary), pad_idx=vocabulary.pad_idx, num_layers=6, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=64, dropout=DROPOUT),
    ImageEncoder(patch_size=16, num_layers=12, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=int(224*224/(16*16)), dropout=DROPOUT, encoder_varient=args.encoder_varient),
    text_dim=384,
    img_dim=384,
    embed_dim=512,
    pad_idx=vocabulary.pad_idx,
    encoder_varient=args.encoder_varient
).to(device)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    import math
    def lr_update(current_step):
        if current_step < num_warmup_steps:
            return float(current_step)/float(max(1, num_warmup_steps))
        return 0.5*(1.0+math.cos(math.pi*float(current_step-num_warmup_steps)/float(max(1, num_training_steps-num_warmup_steps))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
schedular = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*EPOCHS*len(train_loader)), num_training_steps=EPOCHS*len(train_loader))

def training_epoch(model, optimizer, schedular, loader, device):
    model.train()
    loss_epoch = 0;
    for batch in loader:
        X_img, X_text, X_label = batch["img"].to(device), batch["text"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        loss = model(X_img, X_text)
        loss.backward()
        optimizer.step()
        schedular.step()
        loss_epoch+=loss.item()
        break
    return loss_epoch/len(loader)

@torch.inference_mode()
def validation_epoch(model, loader, device):
    model.eval()
    loss_epoch = 0;
    for batch in loader:
        X_img, X_text, X_label = batch["img"].to(device), batch["text"].to(device), batch["label"].to(device)
        loss = model(X_img, X_text)
        loss_epoch+=loss.item()
        break
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
    break
end_time = time.time()
print(f"Time Taken for training: {end_time-start_time}")
