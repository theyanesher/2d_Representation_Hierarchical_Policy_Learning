import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as pl
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference


class MASt3RBackbone(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.model = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.patch_size = 16

        total = sum(p.numel() for p in self.model.parameters())
        print(f"MASt3R params: {total / 1e6:.1f}M (frozen)")

    def _prep(self, img, idx):
        """Convert image(s) to MASt3R input dict.

        Supports both single and batched inputs:
            - Single: (3, H, W) or (H, W, 3) → batch dim added internally
            - Batch:  (B, 3, H, W) or (B, H, W, 3)

        Args:
            img: float [0,1] or uint8 tensor
            idx: view index for MASt3R
        Returns:
            dict with 'img' (B, 3, nH, nW), 'true_shape' (B, 2), 'idx', 'instance'
        """
        if img.dtype == torch.uint8:
            img = img.float() / 255.0

        # Handle both 3D (single) and 4D (batch) inputs
        single = img.ndim == 3
        if single:
            img = img.unsqueeze(0)  # (1, ?, H, W) or (1, H, W, ?)

        # channels-last → channels-first
        if img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        # strip alpha
        if img.shape[1] == 4:
            img = img[:, :3]

        img = (img - 0.5) / 0.5

        B, _, H, W = img.shape
        scale = 512 / max(H, W)
        nH, nW = int(H * scale), int(W * scale)
        img = F.interpolate(img, size=(nH, nW),
                            mode='bilinear', align_corners=False)

        nH = (nH // self.patch_size) * self.patch_size
        nW = (nW // self.patch_size) * self.patch_size
        top = (img.shape[2] - nH) // 2
        left = (img.shape[3] - nW) // 2
        img = img[:, :, top:top + nH, left:left + nW]

        true_shape = np.int32([[nH, nW]] * B)
        return dict(img=img, true_shape=true_shape,
                    idx=idx, instance=str(idx))

    def _run_inference(self, img1, img2):
        v1 = self._prep(img1, idx=0)
        v2 = self._prep(img2, idx=1)
        return inference([(v1, v2)], self.model, self.device,
                         batch_size=1, verbose=False)

    def _reshape_desc(self, desc, true_shape):
        H, W = true_shape[0]
        h, w = H // self.patch_size, W // self.patch_size
        return desc.squeeze(0).reshape(h, w, -1).permute(2, 0, 1)

    @torch.no_grad()
    def forward(self, imgs1, imgs2):
        """
        Args:
            imgs1, imgs2: (B, 3, H, W) float [0, 1]
        Returns:
            feats1, feats2: (B, D, h, w) descriptor maps
        """
        B = imgs1.shape[0]
        v1 = self._prep(imgs1, idx=0)
        v2 = self._prep(imgs2, idx=1)
        pairs = [(v1, v2)]
        output = inference(pairs, self.model, self.device,
                        batch_size=B, verbose=False)
        H, W = output['view1']['true_shape'][0]
        h, w = H // self.patch_size, W // self.patch_size
        f1 = output['pred1']['desc'].reshape(B, h, w, -1).permute(0, 3, 1, 2).to(self.device)
        f2 = output['pred2']['desc'].reshape(B, h, w, -1).permute(0, 3, 1, 2).to(self.device)
        return f1, f2

    def forward_single(self, img1, img2):
        """(3, H, W) inputs — convenience wrapper"""
        f1, f2 = self.forward(img1.unsqueeze(0), img2.unsqueeze(0))
        return f1.squeeze(0), f2.squeeze(0)

    @torch.no_grad()
    def extract_all(self, img1, img2):
        v1 = self._prep(img1, idx=0)
        v2 = self._prep(img2, idx=1)
        output = inference([(v1, v2)], self.model, self.device,
                        batch_size=1, verbose=False)
        f1 = self._reshape_desc(output['pred1']['desc'], output['view1']['true_shape'])
        f2 = self._reshape_desc(output['pred2']['desc'], output['view2']['true_shape'])
        return f1, f2, output

    @torch.no_grad()
    def visualize_matches(self, img1, img2, n_viz=40, save_path='mast3r_matches.png'):
        _, _, output = self.extract_all(img1, img2)

        desc1 = output['pred1']['desc'].squeeze(0).detach()
        desc2 = output['pred2']['desc'].squeeze(0).detach()
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8, device=self.device,
            dist='dot', block_size=2**13)

        H0, W0 = output['view1']['true_shape'][0]
        H1, W1 = output['view2']['true_shape'][0]
        valid = ((matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) &
                 (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3) &
                 (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) &
                 (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3))
        matches_im0, matches_im1 = matches_im0[valid], matches_im1[valid]
        print(f"Total matches: {matches_im0.shape[0]}")

        idx = np.round(np.linspace(0, matches_im0.shape[0] - 1, n_viz)).astype(int)
        viz_m0, viz_m1 = matches_im0[idx], matches_im1[idx]

        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        imgs = []
        for view in [output['view1'], output['view2']]:
            rgb = (view['img'] * std + mean).squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            imgs.append(rgb)

        h0, w0 = imgs[0].shape[:2]
        h1, w1 = imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(h1 - h0, 0)), (0, 0), (0, 0)), constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(h0 - h1, 0)), (0, 0), (0, 0)), constant_values=0)
        canvas = np.concatenate((img0, img1), axis=1)

        pl.figure(figsize=(14, 6))
        pl.imshow(canvas)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_m0[i].T, viz_m1[i].T
            pl.plot([x0, x1 + w0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)),
                    scalex=False, scaley=False)
        pl.axis('off')
        pl.tight_layout()
        pl.show(block=True)

if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms.functional as TF
    import sys

    model = MASt3RBackbone()

    img1 = TF.to_tensor(Image.open('diffusion_policy/diffusion_policy/policy/view_0.png').convert('RGB'))
    img2 = TF.to_tensor(Image.open('diffusion_policy/diffusion_policy/policy/view_1.png').convert('RGB'))

    # --- basic usage (what you'd use in training) ---
    feat1, feat2 = model(img1.unsqueeze(0), img2.unsqueeze(0))
    print(f"feat1: {feat1.shape}, feat2: {feat2.shape}")
    model.visualize_matches(img1, img2)
