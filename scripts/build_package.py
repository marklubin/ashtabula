#!/usr/bin/env python
"""
Build the Python package for distribution.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    """Main function to build the package."""
    print("\n=== Building Ashtabula Package ===\n")
    
    # Clean up previous builds
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # Build source distribution only
    try:
        subprocess.run(
            ["python", "setup.py", "sdist"],
            check=True
        )
        print("\n✓ Package built successfully")
        
        # List the built files
        files = list(Path("dist").glob("*"))
        if files:
            print("\nBuilt files:")
            for file in files:
                print(f"  - {file.name}")
        
        # Check the package
        if Path("dist").exists() and len(list(Path("dist").glob("*"))) > 0:
            try:
                subprocess.run(
                    ["twine", "check", "dist/*"],
                    check=True
                )
                print("\n✓ Package checks passed")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if isinstance(e, FileNotFoundError):
                    print("\n⚠️ Twine not found, skipping package checks")
                    print("  To install: pip install twine")
                else:
                    print("\n✗ Package checks failed")
                    return False
        else:
            print("\n✗ No distribution files found")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Package build failed: {e}")
        return False
    
    print("\n=== Upload Instructions ===\n")
    print("To upload to PyPI, run the following commands:")
    print("\n# Upload to Test PyPI first")
    print("twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
    print("\n# Then upload to the real PyPI")
    print("twine upload dist/*")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)