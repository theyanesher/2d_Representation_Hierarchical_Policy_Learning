import typer
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path

# --- CONFIGURATION ---
REPO_ID = "nakuramino/rgb_articubot"
LOCAL_DIR = "data/"
REPO_TYPE = "dataset"

app = typer.Typer(help="Manage upload/download of the RGB Articubot dataset to Hugging Face.")

@app.command()
def upload(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would happen without uploading.")
):
    """
    Uploads only folders starting with 'rgb' inside LOCAL_DIR to Hugging Face.
    """
    print(f"🚀 Preparing to upload from '{LOCAL_DIR}' to '{REPO_ID}'...")

    if dry_run:
        print("[Dry Run] Would create repo and upload 'rgb*' folders.")
        return

    api = HfApi()
    
    # Create repo if it doesn't exist
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    
    print(f"Uploading 'rgb*' folders...")
    api.upload_large_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=["rgb*/**"], # Only allow files/folders starting with rgb
    )
    print("✅ Upload complete!")

@app.command()
def download(
    force: bool = typer.Option(False, "--force", help="Force redownload even if files exist.")
):
    """
    Downloads the dataset from Hugging Face to LOCAL_DIR.
    """
    print(f"📥 Starting download from '{REPO_ID}' to '{LOCAL_DIR}'...")
    
    # Ensure directory exists
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=LOCAL_DIR,
        resume_download=True,
        max_workers=8,
        force_download=force
    )
    print(f"✅ Download complete! Files are in: {local_path}")

if __name__ == "__main__":
    app()