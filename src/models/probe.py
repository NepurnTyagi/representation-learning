import torch
from torch import nn

class LinearProbe(nn.Module):
    def __init__(self, encoder, embed_dim, output_dim, encoder_varient):
        super().__init__()
        self.encoder = encoder
        for params in self.encoder.parameters():
            params.requires_grad = False
        self.encoder.eval()
        self.linear = nn.Linear(in_features=embed_dim, out_features=output_dim)
        self.encoder_varient=encoder_varient
        self.encode = self.encode_cls if self.encoder_varient=='CLS' else self.encode_gap

    def forward(self, X):
        X = self.encode(X)
        X = self.linear(X)
        return X

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_cls(self, X):
        X = self.encoder(X)
        X = X[:, 0, :]
        return X

    def encode_gap(self, X):
        X = self.encoder(X)
        X = X.mean(dim=1)
        return X
