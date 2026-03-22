#!/usr/bin/env python3
"""Add mast3r feature matches to h5 files in a directory.

For each h5 file:
  1. Load obs/cam0_image[0] and obs/cam1_image[0]
  2. Run MASt3R stereo inference for feature matching
  3. Save matched 2D pixel coordinates as obs/mast3r_matches (static across timesteps)
     Shape: (T, N, 2, 2) — N matches, each with [cam0_xy, cam1_xy]

Coordinates are in original image space (256x256), not MASt3R's resized space.
Idempotent: deletes existing key before writing.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Add project root so imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "mast3r"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "mast3r" / "dust3r"))
sys.path.insert(0, str(PROJECT_ROOT / "diffusion_policy"))

from diffusion_policy.policy.mast3r_backbone import MASt3RBackbone
from mast3r.fast_nn import fast_reciprocal_NNs

H5_KEY = 'obs/mast3r_matches'
NUM_MATCHES = 1500  # fixed match count: crop or zero-pad


def extract_matches(model, cam0_img, cam1_img):
    """Run MASt3R on a stereo pair and return matched pixel coords in original image space.

    Args:
        model: MASt3RBackbone instance
        cam0_img: (H, W, 3) uint8 numpy array
        cam1_img: (H, W, 3) uint8 numpy array

    Returns:
        matches: (N, 2, 2) float32 — matches[:, 0] = cam0 (x,y), matches[:, 1] = cam1 (x,y)
                 coordinates in original image space
        matches_im0_mast3r: (N, 2) in MASt3R space (for visualization)
        matches_im1_mast3r: (N, 2) in MASt3R space (for visualization)
        output: raw mast3r output dict
    """
    orig_H, orig_W = cam0_img.shape[:2]

    img0_t = torch.from_numpy(cam0_img).permute(2, 0, 1).float() / 255.0
    img1_t = torch.from_numpy(cam1_img).permute(2, 0, 1).float() / 255.0

    _, _, output = model.extract_all(img0_t, img1_t)

    desc1 = output['pred1']['desc'].squeeze(0).detach()
    desc2 = output['pred2']['desc'].squeeze(0).detach()

    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2, subsample_or_initxy1=8,
        device=model.device, dist='dot', block_size=2**13
    )

    # Filter border matches in MASt3R space
    H0, W0 = output['view1']['true_shape'][0]
    H1, W1 = output['view2']['true_shape'][0]
    valid = (
        (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) &
        (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3) &
        (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) &
        (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
    )
    matches_im0 = matches_im0[valid]
    matches_im1 = matches_im1[valid]

    # Map MASt3R pixel coords back to original image space
    # MASt3R _prep: scale = 512/max(H,W), center-crop to patch-aligned size
    scale = 512 / max(orig_H, orig_W)
    nH, nW = int(orig_H * scale), int(orig_W * scale)
    nH_crop = (nH // model.patch_size) * model.patch_size
    nW_crop = (nW // model.patch_size) * model.patch_size
    top = (nH - nH_crop) // 2
    left = (nW - nW_crop) // 2

    cam0_orig = np.empty_like(matches_im0, dtype=np.float32)
    cam0_orig[:, 0] = (matches_im0[:, 0] + left) / scale
    cam0_orig[:, 1] = (matches_im0[:, 1] + top) / scale

    cam1_orig = np.empty_like(matches_im1, dtype=np.float32)
    cam1_orig[:, 0] = (matches_im1[:, 0] + left) / scale
    cam1_orig[:, 1] = (matches_im1[:, 1] + top) / scale

    # Stack into (N, 2, 2): [cam0_xy, cam1_xy]
    matches = np.stack([cam0_orig, cam1_orig], axis=1)  # (N, 2, 2)

    # Pad or crop to fixed count
    N = matches.shape[0]
    if N > NUM_MATCHES:
        matches = matches[:NUM_MATCHES]
    elif N < NUM_MATCHES:
        pad = np.zeros((NUM_MATCHES - N, 2, 2), dtype=np.float32)
        matches = np.concatenate([matches, pad], axis=0)

    return matches, matches_im0, matches_im1, output


def visualize_matches(output, matches_im0, matches_im1, n_viz=40, save_path=None):
    """Visualize feature matches on the stereo pair (in MASt3R space)."""
    from matplotlib import pyplot as pl

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

    n_viz = min(n_viz, len(matches_im0))
    if n_viz == 0:
        print("No matches to visualize.")
        return

    idx = np.round(np.linspace(0, len(matches_im0) - 1, n_viz)).astype(int)
    viz_m0, viz_m1 = matches_im0[idx], matches_im1[idx]

    pl.figure(figsize=(14, 6))
    pl.imshow(canvas)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_m0[i].T, viz_m1[i].T
        pl.plot([x0, x1 + w0], [y0, y1], '-+', color=cmap(i / max(n_viz - 1, 1)),
                scalex=False, scaley=False)
    pl.axis('off')
    pl.tight_layout()
    if save_path:
        pl.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved match visualization to {save_path}")
    else:
        pl.show(block=False)
        pl.pause(0.5)
        pl.close()


def process_directory(input_dir, save_viz=False):
    """Process all h5 files in a directory."""
    input_dir = Path(input_dir)
    h5_files = sorted(input_dir.rglob("*.h5"))

    if not h5_files:
        print(f"No h5 files found in {input_dir}")
        return

    print(f"Found {len(h5_files)} h5 files in {input_dir}")

    print("Loading MASt3R model...")
    model = MASt3RBackbone(device='cuda')
    print()

    for i, h5_path in enumerate(h5_files):
        print(f"[{i+1}/{len(h5_files)}] Processing {h5_path.name}")

        with h5py.File(h5_path, 'a') as f:
            if H5_KEY in f:
                del f[H5_KEY]
                print(f"  Deleted existing {H5_KEY}")

            cam0_img = f['obs/cam0_image'][0]  # (H, W, 3) uint8
            cam1_img = f['obs/cam1_image'][0]  # (H, W, 3) uint8
            T = f['obs/cam0_image'].shape[0]

            matches, m0, m1, output = extract_matches(model, cam0_img, cam1_img)

            print(f"  Matches: {matches.shape[0]} correspondences")

            if save_viz:
                viz_save = str(h5_path.with_suffix('.matches.png'))
                visualize_matches(output, m0, m1, n_viz=min(m0.shape[0], 60), save_path=viz_save)

            # Save as static dataset replicated across all timesteps
            # Shape: (T, N, 2, 2) — same matches for every timestep
            static = np.broadcast_to(matches[None], (T,) + matches.shape)
            f.create_dataset(
                H5_KEY,
                data=static,
                compression='gzip',
                compression_opts=4
            )
            print(f"  Saved {H5_KEY} with shape {static.shape}")

        print()

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing h5 files (searched recursively)')
    parser.add_argument('--save-viz', action='store_true',
                        help='Save visualization images next to h5 files')
    args = parser.parse_args()

    process_directory(args.input_dir, save_viz=args.save_viz)
