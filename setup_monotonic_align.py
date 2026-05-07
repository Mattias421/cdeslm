#!/usr/bin/env python3
"""Setup script for compiling Matcha-TTS monotonic_align Cython extension.

This script compiles the monotonic_align.core module which is required for
proper MAS (Monotonic Alignment Search) computation during feature extraction.

Usage:
    python setup_monotonic_align.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_build_dir() -> Path | None:
    """Find the build directory containing the compiled .so file."""
    build_dir = Path(__file__).parent / "Matcha-TTS/matcha/utils/monotonic_align/build"
    if not build_dir.exists():
        return None

    for root, dirs, files in os.walk(build_dir):
        for f in files:
            if f.endswith(".so"):
                return Path(root)
    return None


def copy_so_file(src_dir: Path, dest_dir: Path) -> bool:
    """Copy .so file from build directory to package directory."""
    so_files = list(src_dir.glob("*.so"))
    if not so_files:
        return False

    so_file = so_files[0]
    dest_file = dest_dir / so_file.name
    shutil.copy2(so_file, dest_file)
    print(f"  Copied: {so_file.name} -> {dest_file}")
    return True


def check_cython_installed() -> bool:
    """Check if Cython is installed."""
    import importlib.util

    return importlib.util.find_spec("Cython") is not None


def main():
    script_dir = Path(__file__).parent
    ma_dir = script_dir / "Matcha-TTS/matcha/utils/monotonic_align"

    print("=" * 60)
    print("Matcha-TTS monotonic_align Setup")
    print("=" * 60)

    if not ma_dir.exists():
        print(f"ERROR: monotonic_align directory not found at {ma_dir}")
        sys.exit(1)

    print(f"\nmonotonic_align dir: {ma_dir}")

    existing_so = list(ma_dir.glob("*.so"))
    if existing_so:
        print(f"  .so file already exists: {existing_so[0].name}")
        print("  Verification...")
        try:
            sys.path.insert(0, str(script_dir / "Matcha-TTS"))
            from matcha.utils.monotonic_align import core

            print("  SUCCESS: Module imports correctly!")
            print("\nmonotonic_align is ready to use.")
            return
        except ImportError as e:
            print(f"  WARNING: Existing .so file doesn't work ({e})")
            print("  Rebuilding...")
            existing_so[0].unlink()

    if not check_cython_installed():
        print("\nERROR: Cython is not installed.")
        print("Install it with: pip install Cython")
        sys.exit(1)

    print("\nStep 1: Compiling Cython extension (standalone)...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext"],
        cwd=ma_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("  ERROR: build_ext failed")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    print("  OK: Extension compiled")

    print("\nStep 2: Copying .so file to package directory...")
    build_dir = get_build_dir()
    if build_dir and copy_so_file(build_dir, ma_dir):
        print("  OK: .so file copied")
    else:
        so_files = list(ma_dir.glob("*.so"))
        if so_files:
            print(f"  OK: .so already in place: {so_files[0].name}")
        else:
            print("  WARNING: Could not find .so file")
            build_dir_path = ma_dir / "build"
            if build_dir_path.exists():
                print(f"    Build dir: {build_dir_path}")
                for f in build_dir_path.rglob("*.so"):
                    print(f"    Found: {f}")
            sys.exit(1)

    print("\nStep 3: Verification...")
    try:
        sys.path.insert(0, str(script_dir / "Matcha-TTS"))
        from matcha.utils.monotonic_align import core

        print(f"  Module loaded: {core}")
        print(f"  Has maximum_path_c: {hasattr(core, 'maximum_path_c')}")
        print("\nSUCCESS: monotonic_align is ready!")

    except ImportError as e:
        print(f"  ERROR: Import failed: {e}")
        print("\nManual fix:")
        print(f"  cd {ma_dir}")
        print("  python setup.py build_ext --inplace")
        print("  cp build/lib.linux-*/monotonic_align/core*.so .")
        sys.exit(1)


if __name__ == "__main__":
    main()
