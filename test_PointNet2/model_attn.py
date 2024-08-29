import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D
import numpy as np
import einops
from tqdm import tqdm

class AttnModel(nn.Module):
    def __init__(self, num_classes=13, attn_hidden_dim=120):
        super(AttnModel, self).__init__()
        self.num_classes = num_classes
        
        mlp_hidden_dim = 4 * attn_hidden_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, attn_hidden_dim)
        )

        self.rotary_pos_enc = RotaryPositionEncoding3D(attn_hidden_dim)
        self.attn_layer = RelativeCrossAttentionModule(attn_hidden_dim, 4, 2)


        self.output_mlp = nn.Sequential(
            nn.Linear(attn_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 3, N)
        B, C, N = x.size()
        x = x.permute(0, 2, 1)  # (B, N, 3)
        feat = self.input_mlp(x)
        feat = feat.reshape(B, N, -1)
        feat = einops.rearrange(feat, 'B N D -> N B D')

        pos_embedding = self.rotary_pos_enc(x) # B, N, D, 2
        attn_output = self.attn_layer(query=feat, value=feat, query_pos=pos_embedding, value_pos=pos_embedding)[-1] # N, B, D

        attn_output = einops.rearrange(attn_output, 'N B D -> B N D')
        out = self.output_mlp(attn_output)
        return out


if __name__ == '__main__':
    model = AttnModel()
    model.eval()
    model.cuda()
    for _ in tqdm(range(1000)):
        x = torch.randn(12, 3, 4500).cuda()
        out = model(x)
        print(out.size())
    print(out.size)



            

