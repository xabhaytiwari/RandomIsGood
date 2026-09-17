"""
Lightweight checkpoint backup/restore to a private Hugging Face Hub dataset repo.

WHY THIS EXISTS: Kaggle's free tier caps GPU sessions at ~9-12h and can end a session
without warning (idle timeout, weekly quota exhaustion). /kaggle/working is not
guaranteed to survive across sessions unless you explicitly commit a version. This
module pushes every checkpoint/state/results file to HF Hub the instant it's written,
and pulls everything back down at the start of a fresh session -- so the existing
resume logic in final_script.py picks up exactly where it left off, regardless of
which physical Kaggle session it's running in.

ONE-TIME SETUP:
  1. Create a free account: https://huggingface.co/join
  2. Create a WRITE access token: https://huggingface.co/settings/tokens
  3. In your Kaggle notebook: Add-ons > Secrets > add a secret named HF_TOKEN
  4. Change HF_REPO_ID below to "<your-username>/al-checkpoints" (repo is
     auto-created as private on first push -- no need to create it manually)
  5. pip install huggingface_hub  (usually already present on Kaggle)
"""
import os
import shutil

HF_REPO_ID = "YOUR_USERNAME/al-checkpoints"  # <-- CHANGE THIS to your HF username
HF_REPO_TYPE = "dataset"

_HF_AVAILABLE = True
try:
    from huggingface_hub import HfApi, create_repo, upload_file, list_repo_files, hf_hub_download
except Exception:
    _HF_AVAILABLE = False


def _get_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def ensure_repo():
    if not _HF_AVAILABLE:
        return
    token = _get_token()
    if not token:
        print("  -> [HF Backup] No HF_TOKEN found -- backups disabled for this run.")
        return
    try:
        create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, private=True, token=token, exist_ok=True)
    except Exception as e:
        print(f"  -> [HF Backup] repo setup warning (continuing without backup): {e}")


def push_checkpoint(local_path: str, path_in_repo: str = None):
    """Upload one file, preserving its relative path. Call this immediately after
    ANY checkpoint/state/results file is written locally. Never raises -- a backup
    failure must not crash the actual training run."""
    if not _HF_AVAILABLE or not os.path.exists(local_path):
        return
    token = _get_token()
    if not token:
        return
    path_in_repo = path_in_repo or local_path.lstrip("./")
    try:
        upload_file(
            path_or_fileobj=local_path, path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=token,
            commit_message=f"backup: {path_in_repo}",
        )
        print(f"  -> [HF Backup] pushed {path_in_repo}")
    except Exception as e:
        print(f"  -> [HF Backup] WARNING: push failed for {path_in_repo}: {e}")


def pull_all(local_root: str = "."):
    """Call ONCE at the very start of a fresh session, before main(), to restore
    every previously-backed-up file. Safe to call even if nothing exists yet."""
    if not _HF_AVAILABLE:
        print("  -> [HF Backup] huggingface_hub not installed -- skipping restore.")
        return
    token = _get_token()
    if not token:
        print("  -> [HF Backup] No HF_TOKEN found -- skipping restore (fresh start).")
        return
    try:
        files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=token)
    except Exception as e:
        print(f"  -> [HF Backup] Nothing to restore yet (repo may not exist yet): {e}")
        return
    restored = 0
    for f in files:
        if f in (".gitattributes", "README.md"):
            continue
        local_path = os.path.join(local_root, f)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        try:
            downloaded = hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=f, token=token)
            shutil.copy(downloaded, local_path)
            restored += 1
        except Exception as e:
            print(f"  -> [HF Backup] WARNING: restore failed for {f}: {e}")
    print(f"  -> [HF Backup] Restored {restored} file(s) from previous session(s).")
