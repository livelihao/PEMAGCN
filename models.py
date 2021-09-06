import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import  MultiHeadAttention
import os, sys
import numpy as np
np.set_printoptions(threshold=np.inf)

class PGCAE(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(PGCAE, self).__init__()
        self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
        self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.weight_m = nn.Parameter(torch.Tensor(opt.embed_dim, opt.embed_dim))
        self.attention = MultiHeadAttention(d_model=opt.embed_dim, d_k=100, d_v=100)
        self.fc_aspect = nn.Linear(opt.embed_dim, 100)
        self.convs1 = nn.ModuleList([nn.Conv1d(opt.embed_dim+opt.position_dim, 100, fs) for fs in [3,4,5]])
        self.convs2 = nn.ModuleList([nn.Conv1d(opt.embed_dim+opt.position_dim, 100, fs) for fs in [3,4,5]])
        self.convs3 = nn.ModuleList([nn.Conv1d(opt.embed_dim, 100, fs) for fs in [3, 4, 5]])
        self.fc = nn.Linear(3*100, 3)

    def forward(self, inputs):
        text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
        x = self.text_embed(text)
        aspect = self.text_embed(aspect_text)
        aspect = aspect.sum(1)/aspect.size(1)
        position = self.pos_embed(position_tag)

        x, att = self.attention(x, x, x)
        x = torch.cat((position, x), dim=-1)

        x1 = [F.tanh(conv(torch.transpose(x, 1, 2))) for conv in self.convs1]
        x2 = [F.relu(conv(torch.transpose(x, 1, 2)) + self.fc_aspect(aspect).unsqueeze(2)) for conv in self.convs2]

        x = [i*j for i,j in zip(x1,x2)]


        x0 = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        out = self.fc(x0)
        return out