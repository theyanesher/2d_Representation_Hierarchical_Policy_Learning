"""
Online heatmap rotation augmentation applied in __getitem__.

Applies a random rotation about the image center to every heatmap observation.
Works on (T, C, H, W) float32 arrays with any number of channels C.
All channels at a given timestep receive the same rotation so that
multi-keypoint heatmaps remain internally consistent.

Augmentation is applied independently per timestep (each t gets its own
random draw), which gives more variety when n_obs_steps > 1.

Border fill:
  - Gaussian heatmaps  (type: heatmap):       border_fill=0.0
    Regions outside the original image have no signal → fill with 0.
  - Ghost heatmaps     (type: ghost_heatmap):  border_fill=1.0
    Distance field — regions outside the image are maximally far
    from any keypoint → fill with 1.0 (max normalised distance).
"""

import numpy as np
import cv2


class HeatmapRotationAugmentation:
    """
    Random rotation augmentation for heatmap observations.

    Parameters
    ----------
    enabled : bool
        Master switch. If False, __call__ is a no-op.
    rot_sigma : float
        Std (degrees) of the Gaussian rotation drawn around the image center.
    rot_max : float
        Hard clamp on the absolute rotation angle (degrees).
        Prevents extreme rotations that produce large corner artifacts.
    border_fill : float
        Pixel value used to fill regions that fall outside the original image
        after rotation. Use 0.0 for Gaussian heatmaps and 1.0 for ghost heatmaps.
    p : float
        Probability of applying augmentation at each timestep. 1.0 = always.
    """

    def __init__(
        self,
        enabled: bool = True,
        rot_sigma: float = 3.0,
        rot_max: float = 10.0,
        border_fill: float = 0.0,
        p: float = 1.0,
    ):
        self.enabled = enabled
        self.rot_sigma = rot_sigma
        self.rot_max = rot_max
        self.border_fill = border_fill
        self.p = p

    def __call__(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        heatmap : np.ndarray, shape (T, C, H, W), dtype float32
            Any number of channels C (1, 4, 8, ...).

        Returns
        -------
        np.ndarray, same shape and dtype as input.
        """
        if not self.enabled:
            return heatmap

        T, C, H, W = heatmap.shape
        out = heatmap.copy()
        cx, cy = W / 2.0, H / 2.0

        for t in range(T):
            if np.random.random() > self.p:
                continue

            theta = float(np.clip(
                np.random.normal(0.0, self.rot_sigma),
                -self.rot_max,
                self.rot_max,
            ))

            M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)

            for c in range(C):
                out[t, c] = cv2.warpAffine(
                    heatmap[t, c],
                    M,
                    (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.border_fill,
                )

        return out
