"""
download_mms.py — Download facebook/mms-tts-fas with progress
Run: python download_mms.py
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Downloading facebook/mms-tts-fas...")
print("Model size: ~307 MB  (one-time download)\n")

from huggingface_hub import snapshot_download
import time

t0 = time.time()
try:
    path = snapshot_download(
        repo_id="facebook/mms-tts-fas",
        repo_type="model",
        local_files_only=False,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    elapsed = time.time() - t0
    print(f"\n✅ Download complete in {elapsed:.0f}s")
    print(f"   Saved to: {path}")

    # List downloaded files
    total = 0
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            total += sz
            print(f"   {f:<40} {sz/1024/1024:.1f} MB")
    print(f"\n   Total: {total/1024/1024:.1f} MB")

except Exception as e:
    print(f"\n❌ Download failed: {e}")
    import traceback
    traceback.print_exc()
