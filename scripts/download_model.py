"""
Download the Qwen2.5-1.5B-Instruct GGUF model from Hugging Face.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

# Default Hugging Face direct resolve URL
DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
DEFAULT_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Resolve relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST_DIR = PROJECT_ROOT / "backend" / "models" / "artifacts" / "llm"


def format_bytes(size: float) -> str:
    """Format bytes to human-readable string (MB/GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def download_file(url: str, dest_path: Path, force: bool = False) -> None:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        size = dest_path.stat().st_size
        if size > 100_000_000:  # > 100 MB, likely valid
            print(f"Model file already exists at {dest_path} ({format_bytes(size)}).")
            print("Use --force to re-download.")
            return

    temp_path = dest_path.with_suffix(".tmp")
    print(f"Downloading model from:")
    print(f"  {url}")
    print(f"Destination:")
    print(f"  {dest_path}")
    print("Connecting...")

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BitcoinForensicsDownloader/1.0"
        },
    )

    start_time = time.time()
    try:
        with urlopen(req) as resp, open(temp_path, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            last_print = 0.0

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 0.5 or (total_size and downloaded >= total_size):
                    last_print = now
                    elapsed = now - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        eta = (total_size - downloaded) / speed if speed > 0 else 0
                        print(
                            f"\rProgress: {pct:5.1f}% | {format_bytes(downloaded)} / {format_bytes(total_size)} "
                            f"| {format_bytes(speed)}/s | ETA: {eta:.0f}s",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rDownloaded: {format_bytes(downloaded)} | {format_bytes(speed)}/s",
                            end="",
                            flush=True,
                        )

            print()  # newline after progress

        # Rename temp file to final destination atomically
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)

        total_time = time.time() - start_time
        print(
            f"Successfully downloaded {format_bytes(dest_path.stat().st_size)} "
            f"in {total_time:.1f}s to {dest_path}"
        )

    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Qwen2.5-1.5B-Instruct GGUF model from Hugging Face"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_MODEL_URL,
        help="Model download URL (defaults to Qwen/Qwen2.5-1.5B-Instruct-GGUF)",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_DEST_DIR,
        help=f"Destination directory (defaults to {DEFAULT_DEST_DIR})",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"Target filename (defaults to {DEFAULT_FILENAME})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file if present",
    )

    args = parser.parse_args()
    dest_path = args.dest_dir / args.filename
    download_file(url=args.url, dest_path=dest_path, force=args.force)


if __name__ == "__main__":
    main()
