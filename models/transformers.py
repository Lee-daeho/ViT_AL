import math
from turtle import forward

import torch
import torch.nn as nn
import numpy as np

from einops.layers.torch import Rearrange
from einops import rearrange

class MHA(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout = 0.1):
        super(MHA, self).__init__()
        self.num_heads = num_heads

        self.att_head_size = hidden_size // num_heads

        self.query_embed = nn.Linear(hidden_size, hidden_size)  
        self.key_embed = nn.Linear(hidden_size, hidden_size)    
        self.value_embed = nn.Linear(hidden_size, hidden_size)  

        self.att_dropout = nn.Dropout(dropout)
        self.att_linear = nn.Sequential(
            nn.Lienar(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        query = self.query_embed(x)
        key = self.key_embed(x)
        value = self.value_embed(x)

        query = rearrange(query, 'b p (n h) -> b n p h')    #B * num_patches * hidden_size(num_heads * att_head_size) -> B * num_heads * num_patches * att_head_size
        key = rearrange(key, 'b p (n h) -> b n p h')
        value = rearrange(value, 'b p (n h) -> b n p h')

        attention_scores = torch.matmul(query, key.tranpose(-1,-2)) / math.sqrt(self.att_head_size) #B * num_heads * num_patches * num_patches
        soft_attention_scores = nn.Softmax(attention_scores, dim=-1)

        soft_attention_scores = self.att_dropout(soft_attention_scores)

        out = torch.matmul(soft_attention_scores, value)    #B * num_heads * num_patches * att_head_size
        out = rearrange(out, 'b n p h -> b p (n h)')        #B * num_patches * hidden_size
        out = self.att_linear(out)

        return out
        


class Embeddings(nn.Module):
    def __init__(self, num_patches, img_size, patch_size, hidden_size, patch_embed=2, channels=3, dropout=None):
        super(Embeddings, self).__init__()
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.patch_embed = patch_embed
        self.channels = channels

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))

        if self.patch_embed == 1:
            self.patch_embedding = nn.Sequential(
                Rearrange('b c (h p) (w p) -> b (h w) (p p c)', p=patch_size),
                nn.Linear(patch_size * patch_size * 3 , hidden_size)    #b * num_paches * hidden_size
            )
        elif self.patch_embed == 2:
            self.patch_embedding = nn.Sequential(
                nn.Conv2d(channels, hidden_size, patch_size, patch_size), #b * hidden_size * img_size/patch_size * img_size/patch_size, (img_size/patch_size)^2 = num_patches
                Rearrange('b h w t -> b (w t) h') #b * num_patches * hidden_size 
            )
        
        if not dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        #NEED TO UPDATE HYBRID VERSION

    def forward(self, x):
        batch_size = x.shape[0]     # B * channels * img_size * img_size

        if self.cls_token.shape[0] != batch_size:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        embed = self.patch_embedding(x)
        embed = torch.cat((cls_tokens, embed), dim=1)   #concat cls token
        embed = embed + self.position_embedding         #add position embedding
        
        if not self.dropout:
            embed = self.dropout(embed)

        return embed


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_size, act='gelu', dropout=0.1):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        activation_functions = {'gelu' : nn.GELU(), 'relu' : nn.ReLU()}

        self.act_fn = activation_functions[act]
        self.dropout = nn.Dropout(dropout)

        self.mlp_module = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            self.act_fn(),
            self.dropout(),
            nn.Linear(mlp_size, hidden_size),
            self.dropout()
        )

    def forward(self, x):

        out = self.mlp_module(x)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, mlp_size):
        super(EncoderBlock, self).__init__()
        self.hidden_size = hidden_size

        self.att_norm = nn.LayerNorm(hidden_size)   #B * num_patches * hidden_size
        self.mha = MHA()
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, mlp_size)
    
    def forward(self, x):
        h = x
        out = self.att_norm(x)
        out = self.mha(out)
        out = out + h

        h = out
        out = self.mlp_norm(out)
        out = self.mlp(out)
        out = out + h

        return out


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, mlp_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            layer = EncoderBlock(hidden_size, mlp_size)
            self.layers.append(layer)
    
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        
        return x


class Transformer(nn.Module):
    def __init__(self, img_size, patch_size, num_heads, hidden_size, num_layers, mlp_size):
        super(Transformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.num_patches = self.img_size // self.patch_size

        self.embeddings = Embeddings(self.num_patches, img_size, patch_size, hidden_size)
        self.encoder = Encoder(hidden_size, num_layers, mlp_size)

    def forward(self, x):

        emb = self.embeddings(x)

        enc = self.encoder(emb)

        return enc


class Vit(nn.Module):
    def __init__(self, img_size, num_classes, patch_size, num_heads, hidden_size, num_layers, mlp_size):
        super(Vit, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        assert img_size % patch_size == 0

        self.transformer = Transformer(img_size, patch_size, num_heads, hidden_size, num_layers, mlp_size)
        self.mlp_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        
        out = self.transformer(x)

        out = self.mlp_head(out[:, 0])

        return out