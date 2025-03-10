#!/usr/bin/env python
"""
Verify Ashtabula package by running tests, static analysis, and 
prepare it for PyPI upload.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(message):
    """Print a formatted header message."""
    print(f"\n{BOLD}{YELLOW}{'=' * 70}{RESET}")
    print(f"{BOLD}{YELLOW}{message}{RESET}")
    print(f"{BOLD}{YELLOW}{'=' * 70}{RESET}\n")

def print_success(message):
    """Print a success message."""
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message):
    """Print an error message."""
    print(f"{RED}✗ {message}{RESET}")

def run_command(command, cwd=None, exit_on_error=True):
    """
    Run a shell command and return the result.
    
    Args:
        command: Command to run
        cwd: Directory to run the command in
        exit_on_error: Whether to exit on error
        
    Returns:
        Tuple of (success, output)
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            cwd=cwd
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if exit_on_error:
            print_error(f"Command failed: {command}")
            print(e.stderr)
            sys.exit(1)
        return False, e.stderr

def run_tests():
    """Run all tests with pytest."""
    print_header("Running tests")
    success, output = run_command("uv run pytest", exit_on_error=False)
    
    if success:
        print_success("All tests passed")
    else:
        print_error("Some tests failed")
        print(output)
        return False
    
    return True

def run_static_analysis():
    """Run static analysis tools."""
    print_header("Running static analysis")
    
    # Run mypy
    success_mypy, output_mypy = run_command(
        "uv run mypy ashtabula/", 
        exit_on_error=False
    )
    
    # Try to run ruff if available
    success_ruff, output_ruff = run_command(
        "uv run ruff check ashtabula/",
        exit_on_error=False
    )
    
    # Try to run the static analysis script if it exists
    if os.path.exists("scripts/static_analysis.py"):
        success_static, output_static = run_command(
            "uv run python -m scripts.static_analysis",
            exit_on_error=False
        )
    else:
        success_static, output_static = True, ""
    
    all_success = success_mypy and (success_ruff or "ruff: command not found" in output_ruff)
    
    if all_success:
        print_success("Static analysis passed")
    else:
        print_error("Static analysis found issues")
        if not success_mypy:
            print("\nMyPy issues:")
            print(output_mypy)
        if not success_ruff and "ruff: command not found" not in output_ruff:
            print("\nRuff issues:")
            print(output_ruff)
        return False
    
    return True

def build_package():
    """Build the Python package."""
    print_header("Building package")
    
    # Clean up previous builds
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # Build source distribution and wheel using setup.py directly
    # since we're not using a fully configured pyproject.toml
    success, output = run_command(
        "uv run python -c \"import subprocess; subprocess.run(['python', 'setup.py', 'sdist', 'bdist_wheel'], check=True)\"", 
        exit_on_error=False
    )
    
    if success:
        print_success("Package built successfully")
        # List the built files
        files = list(Path("dist").glob("*"))
        if files:
            print("\nBuilt files:")
            for file in files:
                print(f"  - {file.name}")
    else:
        print_error("Package build failed")
        print(output)
        return False
    
    return True

def check_package():
    """Check the package for PyPI compatibility."""
    print_header("Checking package")
    
    success, output = run_command(
        "uv run twine check dist/*",
        exit_on_error=False
    )
    
    if success:
        print_success("Package checks passed")
    else:
        print_error("Package checks failed")
        print(output)
        return False
    
    return True

def print_upload_instructions():
    """Print instructions for uploading to PyPI."""
    print_header("Upload Instructions")
    print("To upload to PyPI, run the following commands:")
    print(f"\n{BOLD}# Upload to Test PyPI first{RESET}")
    print("uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
    print(f"\n{BOLD}# Then upload to the real PyPI{RESET}")
    print("uv run twine upload dist/*")

def main():
    """Main function to run verification steps."""
    print_header("Ashtabula Package Verification")
    
    # Check if --force flag is passed
    force = "--force" in sys.argv
    
    steps = [
        ("Running tests", run_tests),
        ("Running static analysis", run_static_analysis),
        ("Building package", build_package),
        ("Checking package", check_package),
    ]
    
    all_success = True
    for name, func in steps:
        if not func():
            all_success = False
            print_error(f"Step failed: {name}")
            
            # In non-interactive environments or with --force flag, continue
            if force:
                print(f"Continuing despite failure (--force enabled)")
                continue
                
            try:
                proceed = input(f"\nContinue despite failure in {name}? (y/n): ").lower() == "y"
                if not proceed:
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\nInteractive input not available, stopping.")
                sys.exit(1)
    
    if all_success:
        print_success("\nAll verification steps passed!")
    else:
        print_error("\nSome verification steps failed.")
    
    print_upload_instructions()

if __name__ == "__main__":
    main()