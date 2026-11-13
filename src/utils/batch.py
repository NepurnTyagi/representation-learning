import json
import pandas as pd
import numpy as np
import torch
from collections import deque

class BatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, json_path, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        with open(json_path, 'r') as f:
            df = json.load(f)
        df = df["scenes"]
        df = pd.DataFrame(df)
        df["keys"] = df.objects.apply(lambda samples: tuple(sorted((sample["shape"], sample["size"], sample["material"], sample["color"]) for sample in samples)))
        df["keys"] = df["keys"].combine(self.dataset.df["object_count"], lambda x, y: (y, x))
        self.keys = df["keys"]

    def __iter__(self):
        indices = torch.randperm(len(self.dataset)).tolist()
        queue = deque(indices)
        while queue:
            batch = []
            vis = set()
            temp = deque()
            size = 0
            while queue:
                idx = queue.popleft()
                key = self.keys[idx]
                if key in vis:
                    temp.append(idx)
                else:
                    batch.append(idx)
                    vis.add(key)
                if len(batch) == self.batch_size:
                    yield batch
                    size += 1
                    batch = []
            queue.extend(temp)
            temp = None
            if size == 0:
                while queue:
                    idx = queue.popleft()
                    key = self.keys[idx]
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                if len(batch) != 0:
                    yield batch
                break

    def __len__(self):
        return len(self.dataset)//self.batch_size

if __name__ == "__main__":
    from data import Dataset, collate_fn
    json_path = "../../Dataset/A2part1/A2_dataset/Part_A/train/clevr_train_captions.json"
    img_dir = "../../Dataset/A2part1/A2_dataset/Part_A/train/images"
    dataset = Dataset(json_path=json_path, img_dir=img_dir, task="ClipPretraining")
    json_path = "../../Dataset/A2part1/A2_dataset/Part_A/train/CLEVR_scenes.json"
    batch_sampler = BatchSampler(dataset, json_path = json_path, batch_size = 128)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=lambda batch: collate_fn(batch, task="ClipPretraining", pad_idx=0, context_length=64))
    count = 0
    for batch in dataloader:
        count += 1
        print(count)	
    print("Sucess")
