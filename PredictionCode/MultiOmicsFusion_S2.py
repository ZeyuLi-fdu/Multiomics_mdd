import math
import torch
import torch.nn as nn
from FTTransformer_S1 import FTTransformer
from typing import Literal,Optional

'''
def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
'''
            
_LINFORMER_KV_COMPRESSION_SHARING = Literal['headwise', 'key-value']

class MultiOmicsFusion(nn.Module):
    def __init__(
        self,
        input_dim_g,
        input_dim_p,
        input_dim_m,
        input_dim_c,
        dim,
        dim_out,
        depth,
        heads,
        attn_dropout = 0.,
        ff_dropout = 0.,
        dropout = 0.,
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
        ):
        super(MultiOmicsFusion, self).__init__()  # Inherited from the parent class nn.Module
        self.fc_gen = FTTransformer(num_continuous = input_dim_g, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
        self.fc_pro = FTTransformer(num_continuous = input_dim_p, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
                                   n_tokens = input_dim_p+1, linformer_kv_compression_ratio = linformer_kv_compression_ratio, linformer_kv_compression_sharing = linformer_kv_compression_sharing)
        self.fc_met = FTTransformer(num_continuous = input_dim_m, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
        self.fc_che = FTTransformer(num_continuous = input_dim_c, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout)        
        
        self.fc_mm = nn.Sequential(
            nn.Linear(dim*4, dim*2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        self.classifier = nn.Linear(dim, dim_out)  # 2nd Full-Connected Layer: hidden node -> output

    def forward(self, g_x, p_x, m_x, c_x):
        _,gen_features,_ = self.fc_gen(g_x)
        _,pro_features,_ = self.fc_pro(p_x)
        _,met_features,_ = self.fc_met(m_x)
        _,che_features,_ = self.fc_che(c_x)
        
        mm_features = self.fc_mm(torch.cat([gen_features, pro_features, met_features, che_features], dim=-1))

        logits = self.classifier(mm_features)
        return logits, mm_features