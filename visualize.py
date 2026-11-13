from src.utils import Dataset, collate_fn
from src.models import Retrieval, TextEncoder, ImageEncoder
import torch
import torchvision
import json
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--BATCH_SIZE", type=int, default=16)
parser.add_argument("--DROPOUT", type=float, default=0.1)
parser.add_argument("--dataset_dir", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--momentum_encoder", action='store_true')
parser.add_argument("--pooling_type", type=str, default='cls')
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--encoder_varient", type=str)
args = parser.parse_args()

BATCH_SIZE=args.BATCH_SIZE
DROPOUT=args.DROPOUT
TASK = "ClipPretraining"
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

json_path = args.data_path
img_dir = None if args.dataset_dir is None else f"{args.dataset_dir}/A2_dataset/Part_Aa/Clevr_official/images/val"
img_transform = torchvision.transforms.Resize((224, 224))
val_dataset = Dataset(json_path=json_path, img_dir=img_dir, task=TASK, img_transform=img_transform, tokenizer_path="tokenizer.json")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, task=TASK, pad_idx=0, context_length=64))
vocabulary = val_dataset.getVocabulary()

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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)

encoder.eval()
img = []
with torch.no_grad():
    y_true = []
    for batch in val_loader:
        X_img, X_text, X_label = batch["img"].to(device), batch["text"].to(device), batch["label"].to(device)
        X_img = encoder(X_img)
        X_img = X_img[:, 0, :]
        img.append(X_img)
        y_true.append(X_label)
    img = torch.concat(img, dim=0)
    y_true = torch.concat(y_true, dim=0)
    embed = tsne.fit_transform(img)
    fig, ax = plt.subplots()
    scatter = ax.scatter(embed[:, 0], embed[:, 1], c=y_true, cmap='viridis', s=5, alpha=0.7)
    ax.set_title(f"t-SNE Visualization of model type: {args.model_type}_{"teacher" if args.momentum_encoder else "student"}, pooling type: {args.pooling_type}")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Object Count")
    fig.savefig(args.output_file)
    print("Sucess")
