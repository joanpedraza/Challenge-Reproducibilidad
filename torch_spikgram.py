import math, random, collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
class SkipGramDataset(Dataset):
    def __init__(self, walks, window, node2idx):
        self.pairs = []
        self.node2idx = node2idx
        for walk in walks:
            walk_ids = [node2idx[n] for n in walk]
            for idx, target in enumerate(walk_ids):
                for off in range(1, window + 1):
                    if idx - off >= 0:
                        self.pairs.append((target, walk_ids[idx - off]))
                    if idx + off < len(walk_ids):
                        self.pairs.append((target, walk_ids[idx + off]))
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

# ---------------------------------------------------------------------
class SGNS(nn.Module):
    def __init__(self, num_nodes, emb_dim):
        super().__init__()
        self.in_emb  = nn.Embedding(num_nodes, emb_dim)
        self.out_emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.uniform_(self.in_emb.weight,  -0.5/emb_dim, 0.5/emb_dim)
        nn.init.constant_(self.out_emb.weight, 0)

    def forward(self, targets, contexts, negatives):
        v_t = self.in_emb(targets)
        v_c = self.out_emb(contexts)
        score_pos = torch.sum(v_t * v_c, dim=1)
        loss_pos  = torch.nn.functional.logsigmoid(score_pos)

        v_n = self.out_emb(negatives)
        score_neg = torch.bmm(v_n.neg(), v_t.unsqueeze(2)).squeeze()
        loss_neg  = torch.nn.functional.logsigmoid(score_neg).sum(1)

        return -(loss_pos + loss_neg).mean()

# ---------------------------------------------------------------------
def build_unigram_table(walks, node2idx, table_size=int(1e6), power=0.75):
    counts = np.zeros(len(node2idx))
    for walk in walks:
        for n in walk:
            counts[node2idx[n]] += 1
    probs = counts ** power
    probs /= probs.sum()
    table = np.random.choice(np.arange(len(node2idx)), size=table_size, p=probs)
    return table

# ---------------------------------------------------------------------
def train_skipgram(
    walks,
    node_list,
    emb_dim=64,
    window=10,
    neg_samples=5,
    epochs=1,
    lr=0.025,
    batch_size=1024,
    device='cpu'
):
    node2idx = {n: i for i, n in enumerate(node_list)}
    idx2node = {i: n for n, i in node2idx.items()}
    
    dataset = SkipGramDataset(walks, window, node2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = SGNS(len(node2idx), emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    unigram_table = build_unigram_table(walks, node2idx)
    table_size = len(unigram_table)

    for epoch in range(epochs):
        for targets, contexts in dataloader:
            targets  = targets.to(device)
            contexts = contexts.to(device)
            neg_ids = torch.from_numpy(
                np.random.choice(unigram_table, size=(targets.shape[0], neg_samples))
            ).long().to(device)

            loss = model(targets, contexts, neg_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    vectors = model.in_emb.weight.data.cpu().numpy()
    return {idx2node[i]: vectors[i] for i in range(len(node2idx))}