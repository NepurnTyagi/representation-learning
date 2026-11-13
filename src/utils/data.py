import json
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
from .tokenizer import Tokenizer
import os

CONTEXT_LENGTH = 64

class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, task, img_transform=None, tokenizer_path=None, data_clean=False):
        super().__init__()
        self.img_dir=img_dir
        self.task=task
        self.tokenizer = None
        self.img_transform = img_transform
        with open(json_path, 'r') as file:
            self.df = json.load(file)
        if self.task == 'colors' or self.task == 'count':
            self.df = pd.DataFrame(self.df['examples'])
        else:
            self.df = pd.DataFrame(self.df)
        if data_clean:
            pass
        if self.task == "ClipPretraining":
            if tokenizer_path is not None:
                self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = Tokenizer.from_dataset(self.df.caption)
        if self.task == "DinoPretraining":
            self.global_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224, scale=(0.6, 1), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.04)], p=0.7),
                torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=15)], p=0.3),
                torchvision.transforms.ToTensor()
            ])
            self.local_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(96, scale=(0.4, 0.8), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.04)], p=0.7),
                torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=15)], p=0.3),
                torchvision.transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        fd = os.dup(2)
        with open(os.devnull, "w") as f:
            os.dup2(f.fileno(), 2)
            if self.img_dir is not None:
                img_path = os.path.join(self.img_dir, self.df.iloc[idx]['image_filename'])
            else:
                img_path = self.df.iloc[idx]['image_path']
            img = PIL.Image.open(img_path).convert("RGB")
            os.dup2(fd, 2)
        os.close(fd)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.task == "DinoPretraining":
            X_global, X_local = [self.global_transforms(img) for _ in range(2)], [self.local_transforms(img) for _ in range(8)]
            img = {
                "X_global": X_global,
                "X_local": X_local
            }
        else:
            img = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1)/255.0
        text = None
        label=None
        if self.task == "ClipPretraining":
            text = [self.tokenizer.bos_idx]+self.tokenizer.encode(self.df.iloc[idx]["caption"])+[self.tokenizer.eos_idx]
            label = self.df.iloc[idx].object_count
        elif self.task == "count":
            label = self.df.iloc[idx].object_count
        elif self.task == "colors":
            label = list(map(float, self.df.iloc[idx].multi_hot))
        return {
            "img": img,
            "text": text,
            "label": label
        }

    def __len__(self):
        return len(self.df)

    def getVocabulary(self):
        return self.tokenizer

def collate_fn(batch, task, pad_idx, context_length):
    if task == "DinoPretraining":
        X_global = torch.concat([torch.stack([sample["img"]["X_global"][i] for i in range(2)], dim=0) for sample in batch], dim=0)
        X_local = torch.concat([torch.stack([sample["img"]["X_local"][i] for i in range(8)], dim=0) for sample in batch], dim=0)
        img = (X_global, X_local)
    else:
        img = torch.stack([sample["img"] for sample in batch], dim=0)
    text = torch.empty([])
    label = torch.empty([])
    if task == "ClipPretraining":
        text = []
        for sample in batch:
            token_idx = sample["text"][:context_length]
            token_idx += [pad_idx for _ in range(context_length-len(token_idx))]
            text.append(token_idx)
        text = torch.from_numpy(np.array(text))
    if task != "DinoPretraining":
        label = torch.tensor([sample["label"] for sample in batch])
    return {
        "img": img,
        "text": text,
        "label": label
    }

if __name__ =="__main__":
    print("Start")
    json_path = "../../Dataset/A2_dataset/Part_A/train/clevr_train_captions.json"
    img_dir = "../../Dataset/A2part1/A2_dataset/Part_A/train/images"
    dataset = Dataset(json_path=json_path, img_dir=img_dir, task="ClipPretraining")
    lengths = []
    for i in range(len(dataset)):
        lengths.append(len(dataset[i]["text"]))
    print(f"min: {min(lengths)} | max: {max(lengths)}")
    import sys
    print("Sucess")
    sys.exit()
    json_path = "../../Dataset/A2_dataset/Part_A/train/clevr_train_captions.json"
    img_dir = "../../Dataset/A2_dataset/Part_A/train/images"
    dataset = Dataset(json_path=json_path, img_dir=img_dir, task="ClipPretraining")
    df = dataset.df
    df["count"] = df.image_index.map(df["image_index"].value_counts())
    df = df.reset_index(names="data_idx")
    data = df[df["count"]==2][["data_idx", "image_index"]]
    data.to_csv("../../duplicated.csv", index=False)
    print("End Of Script")
