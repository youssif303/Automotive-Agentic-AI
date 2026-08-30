"""
AutoBrain Lite — Model Downloader

Downloads the quantized LLM (Phi-3-mini GGUF) from Hugging Face.
Default model is Microsoft/Phi-3-mini-4k-instruct-gguf (Q4_K_M version, ~2.2 GB).

Usage:
    python backend/download_model.py
"""

import os
from pathlib import Path
import click
from huggingface_hub import hf_hub_download


@click.command()
@click.option("--repo", default="Microsoft/Phi-3-mini-4k-instruct-gguf", help="HF Repo ID")
@click.option("--filename", default="Phi-3-mini-4k-instruct-q4.gguf", help="GGUF Filename")
@click.option("--output-dir", default="models", help="Destination folder")
def main(repo: str, filename: str, output_dir: str):
    """
    Download a quantized GGUF model from Hugging Face Hub.
    """
    click.echo(f"\n[Model Download] Fetching model from Hugging Face")
    click.echo("=" * 50)
    click.echo(f"  Repo:     {repo}")
    click.echo(f"  File:     {filename}")
    click.echo(f"  Target:   {output_dir}/{filename}")

    # Ensure models directory exists
    os.makedirs(output_dir, exist_ok=True)
    target_path = Path(output_dir) / filename

    if target_path.exists():
        click.echo(f"\n[OK] Model already exists at '{target_path}'. Skipping download.")
        return

    click.echo("\n[INFO] Starting download (approx 2.2 GB)...")
    try:
        # Download file directly to local_dir
        downloaded_path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        click.echo(f"\n[SUCCESS] Model successfully downloaded to: {downloaded_path}")
    except Exception as e:
        click.echo(f"\n[ERROR] Download failed: {str(e)}")
        # Clean up partial download if it exists
        if target_path.exists():
            try:
                target_path.unlink()
            except:
                pass
        os.sys.exit(1)


if __name__ == "__main__":
    main()
