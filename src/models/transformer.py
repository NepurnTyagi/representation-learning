from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, MLP_dim, dropout):
        super().__init__()
        self.LN1 = nn.LayerNorm((embed_dim, ))
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.LN2 = nn.LayerNorm((embed_dim, ))
        self.Linear1 = nn.Linear(in_features=embed_dim, out_features=MLP_dim)
        self.Linear2 = nn.Linear(in_features=MLP_dim, out_features=embed_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self, X, key_padding_mask=None, attn_mask=None):
        X = self.MSA(self.LN1(X), key_padding_mask, attn_mask)+X
        X = self.dropout(self.MLP(self.LN2(X)))+X
        return X

    def MSA(self, X, key_padding_mask, attn_mask):
        return self.attention(X, X, X, need_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask, is_causal=True if attn_mask is not None else False)[0]

    def MLP(self, X):
        return self.Linear2(nn.functional.gelu(self.Linear1(X)))

class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, MLP_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, MLP_dim, dropout) for _ in range(num_layers)
            ]
        )
        self.LN = nn.LayerNorm((embed_dim, ))

    def forward(self, X, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            X = layer(X, key_padding_mask, attn_mask)
        X = self.LN(X)
        return X
