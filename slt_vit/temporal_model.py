"""Temporal modelling over per-frame embeddings, and the projection into mBART's input space.

Per-frame features [B,T,D] -> optional clip-mean normalization -> Conv1d(k=5) + BatchNorm + ReLU + MaxPool(2)
-> [B,(T-4)//2,H] -> Linear + LayerNorm + ReLU -> [B,(T-4)//2,d_model]. With T=32 the encoder sees 14 steps.

Clip-mean normalization subtracts each clip's mean frame feature, keeping what changes during the clip
(movement) and removing what stays constant (signer appearance, background). It helped the ResNet18 model
on held-out signers and is a config option here so its effect on ViT features can be measured.
"""
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden, d_model, kernel=5, pool=2, clip_mean=True, dropout=0.1):
        super().__init__()
        self.clip_mean = clip_mean
        self.conv = nn.Sequential(nn.Conv1d(in_dim, hidden, kernel), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                  nn.MaxPool1d(pool))
        self.proj = nn.Sequential(nn.Linear(hidden, d_model), nn.LayerNorm(d_model), nn.ReLU(inplace=True), nn.Dropout(dropout))
        nn.init.xavier_uniform_(self.proj[0].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, feats):
        if self.clip_mean:
            feats = feats - feats.mean(dim=1, keepdim=True)
        x = self.conv(feats.transpose(1, 2)).transpose(1, 2)
        return self.proj(x)

    @staticmethod
    def output_length(t, kernel=5, pool=2):
        return (t - kernel + 1) // pool
