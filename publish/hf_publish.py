"""Upload a finished model dir plus its model card to a PRIVATE Hugging Face repo.

  python hf_publish.py MODEL_DIR REPO_ID CARD.md [--upload]

Without --upload it only prints what would be sent. Repos are always created private; making one public is a manual
step in the HF settings. Training state, adapters and pickles are never uploaded.
"""
import argparse, json, os, sys
from huggingface_hub import HfApi

IGNORE = ["ckpt/*", "*.pt", "*.pkl", "*.bin", "optim*", "state*", "items_*", "eval/*", "__pycache__/*", "*.log", "COMPLETE"]
NEED = ["config.json", "decider_config.json"]

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("repo_id"); ap.add_argument("card")
ap.add_argument("--upload", action="store_true")
a = ap.parse_args()

card = open(a.card).read()
if "{{" in card:
    sys.exit(f"{a.card} still has a {{{{placeholder}}}}: fill it in first")
files = sorted(f for f in os.listdir(a.model_dir) if os.path.isfile(os.path.join(a.model_dir, f)))
for f in NEED:
    if f not in files:
        sys.exit(f"missing {a.model_dir}/{f}")
if not any(f.endswith(".safetensors") for f in files):
    sys.exit("no .safetensors in model dir")
print("temperature in decider_config.json:", json.load(open(os.path.join(a.model_dir, "decider_config.json"))).get("temperature"))
total = 0
for f in files:
    s = os.path.getsize(os.path.join(a.model_dir, f)); total += s
    print(f"  {s / 1e9:8.3f} GB  {f}")
print(f"  {total / 1e9:8.3f} GB  total (files matching {IGNORE} are skipped)")
if not a.upload:
    print("dry run: nothing uploaded. Re-run with --upload to create the private repo and send these files.")
    sys.exit(0)

api = HfApi()
print("logged in as", api.whoami()["name"])
api.create_repo(a.repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=a.repo_id, folder_path=a.model_dir, ignore_patterns=IGNORE, commit_message="Upload model")
api.upload_file(repo_id=a.repo_id, path_or_fileobj=card.encode(), path_in_repo="README.md", commit_message="Add model card")
try:
    api.upload_file(repo_id=a.repo_id, path_or_fileobj=open(os.path.join(os.path.dirname(a.card), "..", "LICENSE"), "rb"),
                    path_in_repo="LICENSE", commit_message="Add Apache-2.0 licence")
except FileNotFoundError:
    print("LICENSE not found next to the repo: add it by hand")
print("done: https://huggingface.co/" + a.repo_id, "(private)")
