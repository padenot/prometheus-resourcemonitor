#!/usr/bin/env python3
"""
Simple script to download Prometheus exporters to the current directory.
"""

import os
import platform
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


def get_platform_info():
    """Detect platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if machine in ['arm64', 'aarch64']:
        arch = 'arm64'
    elif machine in ['x86_64', 'amd64']:
        arch = 'amd64'
    else:
        arch = 'amd64'  # fallback
    
    return system, arch


def download_exporter(name, url, target_dir):
    """Download and extract an exporter."""
    print(f"Downloading {name}...")
    print(f"  URL: {url}")
    
    # Download to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
        urlretrieve(url, tmp_file.name)
        tmp_path = Path(tmp_file.name)
    
    # Extract
    print(f"Extracting {name}...")
    extract_dir = Path(f"temp_extract_{name}")
    
    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(extract_dir)
    
    # Find the binary
    binary_name = name.replace('_exporter', '-exporter')
    found = False
    
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if name in file and not file.endswith('.md') and not file.endswith('.txt'):
                src_path = Path(root) / file
                # Check if it's a binary (has executable magic bytes)
                with open(src_path, 'rb') as f:
                    magic = f.read(4)
                    # Check for ELF (Linux) or Mach-O (macOS) magic bytes
                    if magic in [b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xca\xfe\xba\xbe']:
                        dest_path = target_dir / name
                        shutil.move(str(src_path), str(dest_path))
                        dest_path.chmod(0o755)
                        print(f"  ✓ Saved to: {dest_path}")
                        found = True
                        break
        if found:
            break
    
    # Clean up
    shutil.rmtree(extract_dir)
    tmp_path.unlink()
    
    if not found:
        print(f"  ✗ Could not find binary in archive")
        return False
    
    return True


def main():
    system, arch = get_platform_info()
    platform_key = f"{system}-{arch}"
    
    print(f"Platform detected: {platform_key}")
    print("=" * 50)
    
    # URLs for exporters
    exporters = {
        "node_exporter": {
            "linux-amd64": "https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz",
            "linux-arm64": "https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-arm64.tar.gz",
            "darwin-amd64": "https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.darwin-amd64.tar.gz",
            "darwin-arm64": "https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.darwin-arm64.tar.gz",
        },
        # Note: process-exporter doesn't have darwin-arm64 builds yet
        "process_exporter": {
            "linux-amd64": "https://github.com/ncabatoff/process-exporter/releases/download/v0.7.10/process-exporter-0.7.10.linux-amd64.tar.gz",
            "darwin-amd64": "https://github.com/ncabatoff/process-exporter/releases/download/v0.7.10/process-exporter-0.7.10.darwin-amd64.tar.gz",
        }
    }
    
    # Create exporters directory
    exporters_dir = Path("./exporters")
    exporters_dir.mkdir(exist_ok=True)
    
    for exporter_name, urls in exporters.items():
        if platform_key in urls:
            url = urls[platform_key]
            success = download_exporter(exporter_name, url, exporters_dir)
            if success and system == 'darwin':
                binary_path = exporters_dir / exporter_name
                print(f"  Removing macOS quarantine from {binary_path}...")
                os.system(f'xattr -d com.apple.quarantine {binary_path} 2>/dev/null')
        else:
            print(f"✗ No {exporter_name} available for {platform_key}")
        print()
    
    print("=" * 50)
    print("\nTo test the exporters manually:")
    print("\n1. Node exporter:")
    print("   ./exporters/node_exporter --web.listen-address=:9100")
    print("   Then visit: http://localhost:9100/metrics")
    print("\n2. Process exporter (Linux only):")
    print("   ./exporters/process_exporter --web.listen-address=:9256")
    print("   Then visit: http://localhost:9256/metrics")
    print("\nPress Ctrl+C to stop the exporters when done.")


if __name__ == "__main__":
    main()