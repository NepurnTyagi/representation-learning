from src.utils import Dataset, collate_fn
from src.models import ImageEncoder, Dino, LinearProbe
import numpy as np
import json
import torch
import torchvision
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--BATCH_SIZE", type=int, default=16)
parser.add_argument("--DROPOUT", type=float, default=0.1)
parser.add_argument("--dataset_dir", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--model_type", type=str)
parser.add_argument("--momentum_encoder", action='store_true')
parser.add_argument("--pooling_type", type=str)
parser.add_argument("--encoder_varient", type=str)
parser.add_argument("--probe_task", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

BATCH_SIZE=args.BATCH_SIZE
DROPOUT=args.DROPOUT
TASK = args.probe_task if args.probe_task=='count' else 'colors'
args.encoder_varient=args.pooling_type.upper()
encoder_path=f"{args.model_dir}/{'Clip' if args.model_type=='clip' else 'Dino'}Pretraining{args.encoder_varient}/best_model.pt"
if args.model_type == 'dino_teacher':
    args.model_type='dino'
    args.momentum_encoder=1
    args.model_path=f"{args.model_dir}/Probe/{'Clip' if args.model_type=='clip' else 'Dino'}{args.encoder_varient}{TASK}_teacher.pt"
elif args.model_type == 'dino_student':
    args.model_type='dino'
    args.model_path=f"{args.model_dir}/Probe/{'Clip' if args.model_type=='clip' else 'Dino'}{args.encoder_varient}{TASK}_student.pt"
else:
    args.model_path=f"{args.model_dir}/Probe/{'Clip' if args.model_type=='clip' else 'Dino'}{args.encoder_varient}{TASK}.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

img_transform = None
if args.model_type == 'clip':
    img_transform = torchvision.transforms.Resize((224, 224))
elif args.model_type == 'dino':
    img_transform = torchvision.transforms.RandomResizedCrop(96, scale=(0.4, 0.8), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
json_path = args.data_path
img_dir = None if args.dataset_dir is None else f"{args.dataset_dir}/A2_dataset/Part_Aa/Clevr_official/images/val"
val_dataset = Dataset(json_path=json_path, img_dir=img_dir, task=TASK, img_transform=img_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, task=TASK, pad_idx=0, context_length=64))
if args.model_type == 'clip':
    encoder = ImageEncoder(patch_size=16, num_layers=12, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=int(224*224/(16*16)), dropout=DROPOUT, encoder_varient=args.encoder_varient)
else:
    encoder = ImageEncoder(patch_size=16, num_layers=12, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=int(224*224/(16*16)), dropout=DROPOUT, encoder_varient=args.encoder_varient)
if args.model_type=='clip':
    encoder_name="ImageEncoder"
else:
    encoder_name="Teacher" if args.momentum_encoder else "Student"
encoder_state_dict = torch.load(encoder_path, map_location=device)['model']
encoder_state_dict = {
    k.replace(f"{encoder_name}.", ""): v for k, v in encoder_state_dict.items() if k.startswith(f"{encoder_name}.")
}
encoder.load_state_dict(encoder_state_dict)
model = LinearProbe(encoder=encoder, embed_dim=384, output_dim=8 if TASK=='colors' else 11, encoder_varient=args.encoder_varient).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device)['model'])

@torch.inference_mode()
def validation_epoch(model, loader, device):
    model.eval()
    y_predict = []
    for batch in loader:
        X_img, X_text, X_label = batch["img"].to(device), batch["text"].to(device), batch["label"].to(device)
        y_pred = model(X_img)
        y_predict.append(y_pred)
    y_predict = torch.concat(y_predict, dim=0)
    return y_predict

y_pred = validation_epoch(model, val_loader, device)
if TASK=='count':
    y_pred = y_pred.argmax(dim=1)
else:
    y_pred = torch.sigmoid(y_pred>0.5).int()
result = {}
with open(args.data_path, 'r') as f:
    df = json.load(f)
df = df['examples']
idx2color = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
for i in range(y_pred.shape[0]):
    if TASK == 'count':
        predictions = int(y_pred[i])
    else:
        for j in range(y_pred.shape[1]):
            predictions = []
            if y_pred[i][j]!=1:
                predictions.append(idx2color[j])
    result[df[i]["image_filename"]] = predictions
with open(args.output_file, 'w') as f:
    json.dump(result, f)
print("sucess")
