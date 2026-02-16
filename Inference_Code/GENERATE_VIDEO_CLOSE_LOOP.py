import cv2
import os
from PIL import Image
import re
import numpy as np

def images_to_video(img_dir, output_path, fps=30):
    # Collect images and sort numerically based on the number after "TRANSFORM"
    files = [
        f for f in os.listdir(img_dir) if f.endswith(".png")
    ]
    def extract_index(filename):
        match = re.search(r'TRANSFORM(\d+)\.png', filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # put unmatched at the end
    files = sorted(files, key=extract_index)

    if len(files) == 0:
        raise ValueError("No .png images found!")

    # Read first image to get size
    first_img = Image.open(os.path.join(img_dir, files[0]))
    W, H = first_img.size

    # Create video writer
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )

    for f in files:
        img_path = os.path.join(img_dir, f)
        img = np.array(Image.open(img_path).convert("RGB"))  # ensure RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)           # convert to BGR for OpenCV
        img = img.astype(np.uint8)

        if img.shape[1] != W or img.shape[0] != H:
            # Resize to match first frame
            img = cv2.resize(img, (W, H))
            print(f"Resized {f} to ({W},{H})")

        writer.write(img)

    writer.release()
    print(f"🎥 Video saved to: {output_path}")


images_to_video(
    img_dir="CLOSE_LOOP/",
    output_path="output_video_CLOSE_LOOP.mp4",
    fps=20
)
