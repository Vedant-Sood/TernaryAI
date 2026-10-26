"""Hugging Face Hub: authentication check, private repo creation, upload, download, sync.

Tokens are never read from or written to this repository. huggingface_hub resolves them from
the HF_TOKEN environment variable or the cache written by `hf auth login`.
Repositories are created private; making one public needs an explicit `--public` flag.
"""

from __future__ import annotations

import os


def whoami() -> dict:
    """Return {"ok", "user", "orgs", "can_write", "error"} without exposing the token."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().whoami()
    except Exception as e:  # not logged in, offline, or no huggingface_hub
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    auth = info.get("auth", {}).get("accessToken", {})
    role = auth.get("role")
    return {"ok": True, "user": info.get("name"), "orgs": [o.get("name") for o in info.get("orgs", [])],
            "token_role": role, "can_write": role in ("write", "admin") or auth.get("fineGrained") is not None}


def create_private_repo(repo_id: str, public: bool = False, exist_ok: bool = True):
    from huggingface_hub import create_repo
    return create_repo(repo_id, repo_type="model", private=not public, exist_ok=exist_ok)


def upload_folder(local_dir: str, repo_id: str, commit_message: str, public: bool = False,
                  ignore=("*.pt", "optim*", "ckpt/*", "*.tmp")):
    from huggingface_hub import HfApi
    api = HfApi()
    create_private_repo(repo_id, public=public)
    info = api.repo_info(repo_id)
    if getattr(info, "private", True) is False and not public:
        raise RuntimeError(f"{repo_id} already exists and is PUBLIC; refusing to upload without --public")
    return api.upload_folder(folder_path=local_dir, repo_id=repo_id, commit_message=commit_message,
                             ignore_patterns=list(ignore))


def download(repo_id: str, local_dir: str, revision: str | None = None):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, revision=revision, local_dir=local_dir)


def sync_checkpoint(run_dir: str, repo_id: str, step: int):
    """Push the newest checkpoint of a run to a private repo branch `checkpoints` (resumable elsewhere)."""
    from huggingface_hub import HfApi
    from ..training.checkpoint import latest_checkpoint
    ck = latest_checkpoint(run_dir)
    if ck is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    api = HfApi()
    create_private_repo(repo_id)
    try:
        api.create_branch(repo_id, branch="checkpoints", exist_ok=True)
    except TypeError:
        api.create_branch(repo_id, branch="checkpoints")
    return api.upload_folder(folder_path=ck, path_in_repo=f"ckpt/{os.path.basename(ck)}", repo_id=repo_id,
                             revision="checkpoints", commit_message=f"checkpoint step {step}")
