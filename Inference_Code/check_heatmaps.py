import cv2
import numpy as np

# Read image (BGR)
img = cv2.imread("/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/outputs_11th_TRAIN_DATASET_ALL_EPISODE_TRAIN_PRETRAINED_RVT_ONLY_FIRST_FRAME/unnormalized_heatmap_images/episode0/camera1/0.png", cv2.IMREAD_UNCHANGED)
import pdb; pdb.set_trace();
# Convert to float and normalize to [0, 1]
img_norm = img.astype(np.float32) / 255.0

# (Optional) do processing here on img_norm

# Convert back to uint8 and save
img_out = (img_norm * 255).clip(0, 255).astype(np.uint8)
cv2.imwrite("output_NORMALIZED_CHECK_HEATMAP.png", img_out)