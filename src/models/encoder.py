import math
import torch
from torch import nn
from .transformer import Transformer

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, num_layers, embed_dim, num_heads, MLP_dim, context_length, dropout):
        super().__init__()
        self.vocabulary_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        self.positional_embedding = nn.parameter.Parameter(torch.randn(context_length, embed_dim)*0.01)
        self.transformer = Transformer(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, MLP_dim=MLP_dim, dropout=dropout)
        self.context_length = context_length
        self.pad_idx = pad_idx

    def forward(self, X):
        X_lengths = (X!=self.pad_idx).sum(dim=-1)
        device = X.device
        key_padding_mask = torch.arange(self.context_length, device=device).unsqueeze(0)>=X_lengths.unsqueeze(1)
        attn_mask = torch.triu(torch.ones((self.context_length, self.context_length), device=device), diagonal=1).bool()
        X = self.vocabulary_embedding(X)+self.positional_embedding.unsqueeze(0)
        X = self.transformer(X, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return X

class ImageEncoder(nn.Module):
    def __init__(self, patch_size, num_layers, embed_dim, num_heads, MLP_dim, context_length, dropout, encoder_varient):
        super().__init__()
        self.x_class = nn.parameter.Parameter(torch.randn(embed_dim)*0.01)
        self.vocabulary_embedding = nn.Linear(in_features=3*patch_size*patch_size, out_features=embed_dim, bias=False)
        self.positional_embedding = nn.parameter.Parameter(torch.randn(context_length+1, embed_dim)*0.01)
        self.transformer = Transformer(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, MLP_dim=MLP_dim, dropout=dropout)
        self.patch_size = patch_size
        self.context_length = context_length
        self.encoder_varient = encoder_varient

    def forward(self, X):
        X = nn.functional.unfold(X, kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size).permute(0, 2, 1)
        X = torch.concat([self.x_class.unsqueeze(0).repeat(X.shape[0], 1).unsqueeze(1), self.vocabulary_embedding(X)], dim=1) + self.get_positional_embedding(X)
        if self.encoder_varient == 'GAP':
            X = X[:, 1:, :]
        X = self.transformer(X)
        return X

    def get_positional_embedding(self, X):
        if (X.shape[1]+1 == self.positional_embedding.shape[0]):
            return self.positional_embedding
        cls_embedding = self.positional_embedding[0, :].unsqueeze(0)
        positional_grid = self.positional_embedding[1:, :]
        positional_grid = positional_grid.reshape(int(math.sqrt(self.positional_embedding.shape[0])), int(math.sqrt(self.positional_embedding.shape[0])), -1).permute(2, 0, 1)
        positional_grid  = nn.functional.interpolate(positional_grid.unsqueeze(0), size=(int(math.sqrt(X.shape[1]+1)), int(math.sqrt(X.shape[1]+1))), mode='bicubic').squeeze(0)
        positional_grid = positional_grid.permute(1, 2, 0).reshape(-1, self.positional_embedding.shape[-1])
        return torch.concat([cls_embedding, positional_grid], dim=0)

