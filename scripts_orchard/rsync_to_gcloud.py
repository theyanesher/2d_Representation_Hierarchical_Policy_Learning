import os
import subprocess
from multiprocessing import Pool, cpu_count

# Configuration
LOCAL_DIR = "/project/flame/yufeiw2/RoboGen-sim2real/data/dp3_demo"
BUCKET_PATH = "gs://cmu-gpucloud-yufeiw2/dp3_demo_165"  # Replace as needed
NUM_PROCESSES = 10

def gcs_file_exists(gcs_path):
    """Returns True if file exists in GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def upload_file(filepath):
    filename = os.path.basename(filepath)
    destination = f"{BUCKET_PATH}/{filename}"
    
    if gcs_file_exists(destination):
        print(f"⏩ Skipped (already exists): {filename}")
        return
    
    try:
        cmd = ["gcloud", "storage", "cp", filepath, destination]
        # print(cmd)
        result = subprocess.run(
            ["gcloud", "storage", "cp", filepath, destination],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[Success] Uploaded: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"[Failure] Failed to upload {filename}: {e.stderr.strip()}")

def main():
    zip_files = [
        os.path.join(LOCAL_DIR, f)
        for f in os.listdir(LOCAL_DIR)
        if f.endswith(".zip") and os.path.isfile(os.path.join(LOCAL_DIR, f))
    ]
    
    zip_files = sorted(zip_files)

    if not zip_files:
        print("No zip files found.")
        return

    with Pool(processes=min(NUM_PROCESSES, cpu_count())) as pool:
        pool.map(upload_file, zip_files)

if __name__ == "__main__":
    main()
