import torch
from torch import nn
from .transformer import Transformer

class Retrieval(nn.Module):
    def __init__(self, TextEncoder, ImageEncoder, text_dim, img_dim, embed_dim, pad_idx, encoder_varient):
        super().__init__()
        self.TextEncoder = TextEncoder
        self.ImageEncoder = ImageEncoder
        self.img_projection = nn.Linear(in_features=img_dim, out_features=embed_dim, bias=False)
        self.text_projection = nn.Linear(in_features=text_dim, out_features=embed_dim, bias=False)
        self.temperature = nn.parameter.Parameter(torch.log(torch.tensor(1/0.07)))
        self.pad_idx = pad_idx
        self.loss = nn.CrossEntropyLoss()
        self.encoder_varient = encoder_varient
        self.encode_image = self.encode_image_cls if self.encoder_varient=='CLS' else self.encode_image_gap

    def forward(self, X_img, X_text):
        n = X_img.shape[0]
        X_img, X_text = nn.functional.normalize(self.img_projection(self.encode_image(X_img))), nn.functional.normalize(self.text_projection(self.encode_text(X_text)))
        return X_img, X_text

    def encode_image_cls(self, X):
        X = self.ImageEncoder(X)
        X = X[:, 0, :]
        return X

    def encode_image_gap(self, X):
        X = self.ImageEncoder(X)
        X = X.mean(dim=1)
        return X

    def encode_text(self, X):
        X_lengths = (X!=self.pad_idx).sum(dim=-1)
        X = self.TextEncoder(X)
        X = torch.gather(input=X, dim=1, index=(X_lengths-1).unsqueeze(1).unsqueeze(2).repeat(1, 1, X.shape[-1])).squeeze(1)
        return X
