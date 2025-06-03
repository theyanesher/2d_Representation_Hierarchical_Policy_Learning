import torch
from torch import nn
from torchvision.ops import MLP

from test_PointNet2.ptv3.model import Point, PointTransformerV3

ptv3_config = {
    "in_channels": 3,
    "order": ["z", "z-trans", "hilbert", "hilbert-trans"],
    "stride": [2, 2, 2, 2],
    "enc_depths": [2, 2, 2, 6, 2],
    "enc_channels": [32, 64, 128, 256, 512],
    "enc_num_head": [2, 4, 8, 16, 32],
    "enc_patch_size": [128, 128, 128, 128, 128],
    "dec_depths": [2, 2, 2, 2],
    "dec_channels": [64, 64, 128, 256],
    "dec_num_head": [4, 4, 8, 16],
    "dec_patch_size": [128, 128, 128, 128],
    "mlp_ratio": 4,
    "qkv_bias": True,
    "qk_scale": None,
    "attn_drop": 0.0,
    "proj_drop": 0.0,
    "drop_path": 0.3,
    "shuffle_orders": True,
    "pre_norm": True,
    "enable_rpe": False,
    "enable_flash": True,
    "upcast_attention": False,
    "upcast_softmax": False,
    "cls_mode": False,
    "pdnorm_bn": False,
    "pdnorm_ln": False,
    "pdnorm_decouple": True,
    "pdnorm_adaptive": False,
    "pdnorm_affine": True,
    "pdnorm_conditions": ["ScanNet", "S3DIS", "Structured3D"]
}

head_mlp_config = {
    "hidden_channels": [64]
}


class HighlevelPTv3(nn.Module):
    def __init__(self, num_classes, grid_size=0.02):
        super().__init__()
        self.grid_size = grid_size
        head_mlp_config["hidden_channels"].append(num_classes)
        self.ptv3 = PointTransformerV3(**ptv3_config)
        self.mlp_head = MLP(in_channels=self.ptv3.get_out_channels(), **head_mlp_config)

    def forward(self, x):
        # x: BxCxN tensor, where B is batch size, C is number of channels, and N is number of points
        x = x.permute(0, 2, 1)  # BxNxC

        B, N, C = x.shape
        # print(f"Input shape: {B}x{N}x{C}")
        # form data_dict
        offset = torch.arange(1, B + 1) * N
        data_dict = {
            "feat": x.reshape(-1, C),
            "coord": x[..., :3].reshape(-1, 3),
            "grid_size": self.grid_size,
            "offset": offset.to(x.device),
        }
        point = self.ptv3.forward(data_dict)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.mlp_head.forward(feat)
        seg_logits = seg_logits.reshape(B, N, -1)
        # print(f"Output shape: {seg_logits.shape}")
        return seg_logits
