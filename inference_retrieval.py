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
parser.add_argument("--pooling_type", type=str, default='cls')
parser.add_argument("--retrieval_task", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--encoder_varient", type=str)
parser.add_argument("--topk", type=int, default=3)
args = parser.parse_args()

BATCH_SIZE=args.BATCH_SIZE
DROPOUT=args.DROPOUT
TASK = "ClipPretraining"
args.encoder_varient=args.pooling_type.upper()
args.model_path=f"{args.model_dir}/{'Clip' if args.model_type=='clip' else 'Dino'}Pretraining{args.encoder_varient}/best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

json_path = args.data_path
img_dir = None if args.dataset_dir is None else f"{args.dataset_dir}/A2_dataset/Part_Aa/Clevr_official/images/val"
img_transform = torchvision.transforms.Resize((224, 224))
val_dataset = Dataset(json_path=json_path, img_dir=img_dir, task=TASK, img_transform=img_transform, tokenizer_path="tokenizer.json")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, task=TASK, pad_idx=0, context_length=64))
vocabulary = val_dataset.getVocabulary()

model = Retrieval(
    TextEncoder(vocab_size=len(vocabulary), pad_idx=vocabulary.pad_idx, num_layers=6, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=64, dropout=DROPOUT),
    ImageEncoder(patch_size=16, num_layers=12, embed_dim=384, num_heads=6, MLP_dim=1536, context_length=int(224*224/(16*16)), dropout=DROPOUT, encoder_varient=args.encoder_varient),
    text_dim=384,
    img_dim=384,
    embed_dim=512,
    pad_idx=vocabulary.pad_idx,
    encoder_varient=args.encoder_varient
).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device)["model"])
model.eval()
img, text = [], []
with torch.no_grad():
    for batch in val_loader:
        X_img, X_text, X_label = batch["img"].to(device), batch["text"].to(device), batch["label"].to(device)
        X_img, X_text = model(X_img, X_text)
        img.append(X_img)
        text.append(X_text)
    img = torch.concat(img, dim=0)
    text = torch.concat(text, dim=0)
    M = img@text.T
    _, indices = torch.topk(M, args.topk, dim=1 if args.retrieval_task == "i2t" else 0)
    if args.retrieval_task == "t2i":
        indices = indices.T
    score = 0
    for i in range(indices.shape[0]):
        if (i==indices[i]).any():
            score+=1
    print(f"Recall Score for {args.topk}, {args.retrieval_task}: {score/indices.shape[0]}")
    result = {}
    with open(args.data_path, 'r') as f:
        df = json.load(f)
    source, target = ('image_filename', 'caption') if args.retrieval_task == "i2t" else ('caption', 'image_filename')
    for i in range(indices.shape[0]):
        predictions = []
        for j in range(args.topk):
            predictions.append(df[indices[i][j]][target])
        result[df[i][source]] = predictions
    with open(args.output_file, 'w') as f:
        json.dump(result, f)
    print("sucess")
