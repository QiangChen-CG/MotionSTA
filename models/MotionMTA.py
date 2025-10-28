import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import torch_dct as dct

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class PositionEncoding(nn.Module):
    def __init__(self, latent_dim, max_len = 5000):
        super().__init__()
        device = torch.device('cuda:0')
        self.pe = torch.zeros(max_len, latent_dim).to(device)
        position = torch.arange(0, max_len, dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-math.log(10000.0) / latent_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('PE', self.pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), : ]
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class StylizationBlock(nn.Module):
    def __init__(self, latent_dim, time_emb_dim):
        super().__init__()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        emb_out = self.emb_layers(emb).unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim = 2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class MotionAttn(nn.Module):
    def __init__(self, latent_dim, num_heads, drop=0.0):
        super().__init__()

        self.conv1 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.act2 = nn.SiLU(())
        self.conv2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(latent_dim, latent_dim * 2)
        self.linear2 = nn.Linear(latent_dim * 2, latent_dim)
        self.m_attn = MHSA_M(latent_dim, num_heads)
        self.ln1 = nn.LayerNorm(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)
        self.ln3 = nn.LayerNorm(latent_dim)
        self.ln4 = nn.LayerNorm(latent_dim)
        self.mlp1 = MLP(latent_dim, latent_dim * 2, latent_dim, drop)
        self.mlp2 = MLP(latent_dim, latent_dim * 2, latent_dim, drop)


    def forward(self, x):

        y = self.m_attn(self.ln1(x)) + x
        d_y = dct.dct(self.ln2(y))
        d_y = self.linear2(self.act1(self.linear1(d_y)))
        d_y = dct.idct(d_y)
        y = self.mlp1(d_y) + self.mlp2(self.ln3(y)) + y
        y = self.conv2(self.act2(self.conv1(self.ln4(y).permute(0, 2, 1)))).permute(0, 2, 1) + y
        #y = F.sigmoid(y)
        return y
    
class TemporalAttn(nn.Module):
    def __init__(self, latent_dim, num_heads, drop=0.0):
        super().__init__()
        
        self.tmp_attn = MHSA(latent_dim, num_heads)
        self.tmp_attn_layer1_1 = MHSA(latent_dim, num_heads)
        self.tmp_attn_layer1_2 = MHSA(latent_dim, num_heads)
        self.tmp_attn_layer1_3 = MHSA(latent_dim, num_heads)
        self.tmp_attn_layer1_4 = MHSA(latent_dim, num_heads)
        self.tmp_attn_layer2_1 = MHSA(latent_dim, num_heads)
        self.tmp_attn_layer2_2 = MHSA(latent_dim, num_heads)
        self.tmp_attn_layer3 = MHSA(latent_dim, num_heads)
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.ln1 = nn.LayerNorm(latent_dim)
        self.tmp_pos_embed_layer1 = nn.Parameter(torch.ones(1, latent_dim))
        self.fusion = nn.Linear(latent_dim, latent_dim)
        self.tmp_conv = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, x):
        #layer1
        x_ = self.tmp_attn(x)
        x_layer1_1, x_layer1_2, x_layer1_3 = torch.chunk(x, 3, dim = 1)
        tmp_layer1_1 = self.tmp_attn_layer1_1(x_layer1_1)
        tmp_layer1_2 = self.tmp_attn_layer1_2(x_layer1_2)
        tmp_layer1_3 = self.tmp_attn_layer1_3(x_layer1_3)

        tmp_layer1 = torch.cat((tmp_layer1_1, tmp_layer1_2, tmp_layer1_3), dim = 1) + x_

        #layer2
        tmp_layer2 = self.tmp_attn_layer2_1(self.ln1(tmp_layer1)) + x_

        #Fusion
        alpha = torch.cat((tmp_layer1.unsqueeze(0), tmp_layer2.unsqueeze(0)), dim = 0)
        alpha = self.fusion(alpha)
        alpha = alpha.softmax(dim = 0)
        out = tmp_layer1 * alpha[0] + tmp_layer2 * alpha[1]

        return out
    
class MHSA(nn.Module):
    def __init__(self, latent_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(latent_dim, latent_dim)
        self.v = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        # B, H, D // H, T
        q = self.q(x).view(B, H, -1, T)
        k = self.k(x).view(B, H, -1, T)
        v = self.v(x).view(B, H, -1, T)

        attn = (k.transpose(-2, -1) @ v) * self.temperature
        attn = attn.softmax(dim = -1)

        out = q @ attn
        out = out.reshape(B, T, D)
        return out

class MHSA_M(nn.Module):
    def __init__(self, latent_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(latent_dim, latent_dim)
        self.v = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        # B, T, H, D // H
        q = self.q(x).view(B, H, -1, T)
        k = self.k(x).view(B, H, -1, T)
        v = self.v(x).view(B, H, -1, T)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim = -1)

        out = attn @ v
        out = out.reshape(B, T, D)
        return out
    
class CA(nn.Module):
    def __init__(self, latent_dim, text_dim, time_emb_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(text_dim, latent_dim)
        self.v = nn.Linear(text_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.proj_out = StylizationBlock(latent_dim, time_emb_dim)

    def forward(self, x, xf, emb):
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_heads

        # B, T, D
        q = self.q(self.norm(x))
        # B, N, D
        k = self.k(self.text_norm(xf))
        q = F.softmax(q.view(B, T, H, -1), dim=-1)
        k = F.softmax(k.view(B, N, H, -1), dim=-1)
        v = self.v(self.text_norm(xf)).view(B, N, H, -1)

        attn = torch.einsum('bnhd,bnhl->bhdl', k, v)

        out = torch.einsum('bnhd,bhdl->bnhl', q, attn).reshape(B, T, D)
        out = x + self.proj_out(out, emb)
        return out
    
class MotionMTA(nn.Module):
    def __init__(self, latent_dim, text_dim, time_emb_dim, tmp_dim, num_heads, drop):
        super().__init__()
        
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.ca = CA(latent_dim, text_dim, time_emb_dim, num_heads)
        self.tmp_attn = TemporalAttn(latent_dim, num_heads, drop)
        self.motion_attn = MotionAttn(latent_dim, num_heads, drop)
        self.conv = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.mlp1 = MLP(latent_dim, latent_dim * 2, latent_dim, drop)
        self.ln1 = nn.LayerNorm(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)
        self.ln3 = nn.LayerNorm(latent_dim)
        self.sty_proj1 = StylizationBlock(latent_dim, time_emb_dim)
        self.sty_proj2 = StylizationBlock(latent_dim, time_emb_dim)
        self.sty_proj3 = StylizationBlock(latent_dim, time_emb_dim)


    def forward(self, x, xf, emb):
        y = self.ca(self.ln1(x), xf, emb)

        T_A = self.sty_proj1(self.tmp_attn(y), emb) + y

        M_A = self.sty_proj2(self.motion_attn(T_A), emb) + T_A

        out = self.sty_proj3(self.mlp1(self.ln2(M_A)), emb) + M_A

        return out