# # import pickle
# # from PIL import Image
# # import numpy as np
# # import os

# # # 1️⃣ Load the pickle file
# # with open("DUMP_INPUT_TO_NETWORK.pkl", "rb") as f:
# #     data = pickle.load(f)
# # # import pdb; pdb.set_trace();
# # # 2️⃣ Extract the part you want as an image
# # # Suppose the dictionary has a key "heatmap" that is a 2D numpy array
# # image_inputs = np.array(data["image"][0][0].detach().cpu()) # shape: (H, W)

# # # If the array is not uint8, convert it
# # # if image_inputs .dtype != np.uint8:
# # #     image_inputs  = (255 * (image_inputs  - image_inputs .min()) / (image_inputs .ptp() + 1e-8)).astype(np.uint8)

# # # 3️⃣ Save as an image
# # # image = Image.fromarray(heatmap_array)
# # # image.save("output/heatmap_image.png")
# # save_dir = "output_images"
# # os.makedirs(save_dir, exist_ok=True)

# # # 1️⃣ Save first 3 RGB images (3 channels each)
# # for i in range(3):
# #     rgb = image_inputs[i*3:(i+1)*3, :, :]  # shape: (3, 224, 224)
# #     rgb_np = (rgb * 255).astype(np.uint8)  # scale to uint8
# #     rgb_np = np.transpose(rgb_np, (1, 2, 0))       # (H, W, C) for PIL
# #     Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_image_{i+1}.png"))

# # # 2️⃣ Save next 3 heatmaps (single-channel)
# # for i in range(3):
# #     heatmap = image_inputs[9 + i, :, :]                # shape: (224, 224)
# #     # Normalize to [0, 255] for visualization
# #     heatmap_np = (255 * (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)).astype(np.uint8)
# #     # heatmap_np = heatmap
# #     Image.fromarray(heatmap_np).save(os.path.join(save_dir, f"heatmap_{i+1}.png"))

# # # for i in range(3):
# # #     # --- RGB image ---
# # #     rgb = image_inputs[i*3:(i+1)*3, :, :]       # shape (3,H,W)
# # #     rgb = np.transpose(rgb, (1, 2, 0))          # (H,W,3)
# # #     rgb_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)

# # #     # --- Heatmap ---
# # #     heatmap = image_inputs[9 + i, :, :]
# # #     heatmap_norm = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)  # scale to [0,1]

# # #     # Lighten heatmap (make it subtle)
# # #     heatmap_light = heatmap_norm * 0.2

# # #     # Convert to uint8 greyscale
# # #     heatmap_uint8 = (heatmap_light * 255).astype(np.uint8)

# # #     # --- Overlay ---
# # #     overlay = rgb_uint8.copy()
# # #     # Add heatmap to each RGB channel equally (greyscale overlay)
# # #     overlay = np.clip(overlay + heatmap_uint8[:, :, None], 0, 255).astype(np.uint8)

# # #     # --- Save overlay ---
# # #     Image.fromarray(overlay).save(os.path.join(save_dir, f"overlay_{i+1}.png"))

# # # for i in range(3):
# # #     # --- RGB image ---
# # #     rgb = image_inputs[i*3:(i+1)*3, :, :]       # shape (3,H,W)
# # #     rgb = np.transpose(rgb, (1, 2, 0))          # (H,W,3), still float [0,1]

# # #     # --- Heatmap ---
# # #     heatmap = image_inputs[9 + i, :, :]
# # #     heatmap_norm = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)  # [0,1]

# # #     # --- Overlay emphasizing high-prob regions ---
# # #     # Multiply heatmap by a factor to exaggerate high values
# # #     alpha = 0.4  # overall strength of heatmap overlay
# # #     overlay = rgb + alpha * heatmap_norm[:, :, None]  # broadcast over RGB channels
# # #     overlay = np.clip(overlay, 0, 1)                 # keep in [0,1]

# # #     # Convert to uint8 for saving
# # #     overlay_uint8 = (overlay * 255).astype(np.uint8)
# # #     Image.fromarray(overlay_uint8).save(os.path.join(save_dir, f"overlay_{i+1}.png"))

# # for i in range(3):
# #     # --- RGB image ---
# #     rgb = image_inputs[i*3:(i+1)*3, :, :]       # shape (3,H,W)
# #     rgb = np.transpose(rgb, (1, 2, 0))          # (H,W,3), float [0,1]

# #     # --- Heatmap ---
# #     heatmap = image_inputs[9 + i, :, :]
# #     heatmap_norm = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)  # [0,1]

# #     # --- Invert heatmap ---
# #     heatmap_inverted = 1.0 - heatmap_norm

# #     # --- Overlay emphasizing the inverted heatmap ---
# #     alpha = 0.5
# #     overlay = rgb + alpha * heatmap_inverted[:, :, None]  # broadcast to RGB
# #     overlay = np.clip(overlay, 0, 1)

# #     # --- Convert to uint8 and save ---
# #     overlay_uint8 = (overlay * 255).astype(np.uint8)
# #     Image.fromarray(overlay_uint8).save(os.path.join(save_dir, f"overlay_inverted_{i+1}.png"))



# import pickle
# import numpy as np
# import os
# import cv2

# # 1️⃣ Load the pickle file
# with open("DUMP_INPUT_TO_NETWORK.pkl", "rb") as f:
#     data = pickle.load(f)

# # 2️⃣ Extract tensor: shape (16, 12, 32, 32)
# tensor = np.array(data["image"][0].detach().cpu())  # adjust indexing if needed
# # e.g., tensor = np.array(data["image"][0][0].detach().cpu())

# num_timesteps, _, H, W = tensor.shape
# save_dir = "output_videos"
# os.makedirs(save_dir, exist_ok=True)

# fps = 4
# alpha = 0.0  # overlay strength

# # --- Create a video for each camera (3 RGB groups) ---
# for cam in range(3):
#     video_path = os.path.join(save_dir, f"camera_{cam+1}.mp4")
#     video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

#     for t in range(num_timesteps):
#         image_inputs = tensor[t]  # shape (12, H, W)

#         # --- RGB for this camera ---
#         rgb = image_inputs[cam*3:(cam+1)*3, :, :]          # shape (3,H,W)
#         rgb = np.transpose(rgb, (1,2,0))                   # (H,W,3)
#         rgb_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)

#         # --- Corresponding heatmap ---
#         heatmap = image_inputs[9 + cam, :, :]             # single-channel
#         heatmap_norm = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)
#         heatmap_inverted = 1.0 - heatmap_norm

#         # --- Overlay ---
#         overlay = np.clip(rgb + alpha * heatmap_inverted[:, :, None], 0, 1)
#         overlay_uint8 = (overlay * 255).astype(np.uint8)

#         # Convert RGB → BGR for OpenCV
#         frame = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
#         video_writer.write(frame)

#     video_writer.release()
#     print(f"Saved video for camera {cam+1} at {video_path}")


import pickle
import numpy as np
import os
import cv2
from PIL import Image

# 1️⃣ Load the pickle file
with open("DUMP_INPUT_TO_NETWORK.pkl", "rb") as f:
    data = pickle.load(f)

# 2️⃣ Extract tensor: shape (T, 12, H, W)
tensor = np.array(data["image"][0].detach().cpu())  # adjust indexing if needed
num_timesteps, _, H, W = tensor.shape

save_dir = "output_videos"
os.makedirs(save_dir, exist_ok=True)

fps = 4
alpha = 0.5  # overlay strength for visualization

# --- Create a video and save individual frames for each camera ---
for cam in range(3):
    cam_dir = os.path.join(save_dir, f"camera_{cam+1}")
    os.makedirs(cam_dir, exist_ok=True)

    # Create video writer
    video_path = os.path.join(cam_dir, f"camera_{cam+1}.mp4")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    for t in range(num_timesteps):
        image_inputs = tensor[t]  # (12, H, W)

        # --- RGB for this camera ---
        rgb = image_inputs[cam*3:(cam+1)*3, :, :]  # (3, H, W)
        rgb = np.transpose(rgb, (1, 2, 0))         # (H, W, 3)
        rgb_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)

        # --- Heatmap ---
        heatmap = image_inputs[9 + cam, :, :]
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)
        heatmap_inverted = 1.0 - heatmap_norm  # make bright = high probability
        heatmap_inverted = heatmap
        # --- Overlay ---
        overlay = np.clip(rgb + alpha * heatmap_inverted[:, :, None], 0, 1)
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        # Convert to BGR for OpenCV video writing
        frame = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

        # --- Save individual images ---
        Image.fromarray(rgb_uint8).save(os.path.join(cam_dir, f"rgb_t{t+1:02d}.png"))
        heatmap_img = (heatmap_inverted * 255).astype(np.uint8)
        Image.fromarray(heatmap_img).save(os.path.join(cam_dir, f"heatmap_t{t+1:02d}.png"))
        Image.fromarray(overlay_uint8).save(os.path.join(cam_dir, f"overlay_t{t+1:02d}.png"))

    video_writer.release()
    print(f"✅ Saved video and frames for camera {cam+1} at {cam_dir}")
