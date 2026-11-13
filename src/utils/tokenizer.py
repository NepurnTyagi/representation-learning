import re
import json

class Tokenizer:
    def __init__(self, vocab):
        self.token2idx = vocab["token2idx"]
        self.idx2token = vocab["idx2token"]
        self.pad_token, self.pad_idx = "<pad>", 0 
        self.unk_token, self.unk_idx = "<unk>", 1        
        self.bos_token, self.bos_idx = "<bos>", 2
        self.eos_token, self.eos_idx = "<eos>", 3
        
    @classmethod
    def from_dataset(cls, tokens):
        bag_of_words = set()
        for token in tokens:
            bag_of_words.update(cls.tokenize(token))
        token2idx = {}
        idx2token = [0, 1, 2, 3]
        idx2token[0], token2idx["<pad>"] = "<pad>", 0
        idx2token[1], token2idx["<unk>"] = "<unk>", 1
        idx2token[2], token2idx["<bos>"] = "<bos>", 2
        idx2token[3], token2idx["<eos>"] = "<eos>", 3
        for token in sorted(bag_of_words):
            token2idx[token] = len(idx2token)
            idx2token.append(token)
        vocab = {
            "token2idx": token2idx,
            "idx2token": idx2token
        }
        return cls(vocab)

    @classmethod
    def from_pretrained(cls, path):
        with open(path, 'r') as f:
            vocab = json.load(f)
        token2idx = vocab["token2idx"]
        idx2token = vocab["idx2token"]
        vocab = {
            "token2idx": token2idx,
            "idx2token": idx2token
        }
        return cls(vocab)

    def save_pretrained(self, path):
        vocab = {
            "token2idx": self.token2idx,
            "idx2token": self.idx2token
        }
        with open(path, "w") as f:
            json.dump(vocab, f)
        return

    @staticmethod
    def tokenize(tokens):
        return re.findall(r"\w+", tokens.lower())

    def detokenize(self, tokens):
        return " ".join(tokens)

    def encode(self, tokens):
        tokens = Tokenizer.tokenize(tokens)
        return [self.token2idx[token] if token in self.token2idx.keys() else self.unk_token for token in tokens]

    def decode(self, indices, skip_special_token=False):
        return [self.idx2token[idx] if not skip_special_token and idx in [0, 2, 3] else "" for idx in indices]

    def __len__(self):
        return len(self.idx2token)

if __name__ == "__main__":
    import pandas as pd
    with open("../../Dataset/A2_dataset/Part_A/train/clevr_train_captions.json", 'r') as file:
        df = json.load(file)
    df=pd.DataFrame(df)
    tokenizer=Tokenizer.from_dataset(df.caption)
    tokenizer.save_pretrained("../../tokenizer.json")
