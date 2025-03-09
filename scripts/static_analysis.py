import subprocess

def run_command(description: str, command: str) -> None:
    """Runs a shell command and prints the output."""
    print(f"\n🔍 Running {description}...\n{'=' * 40}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

def run_static_analysis() -> None:
    """Run all static analysis checks for the project."""
    print("\n🚀 Running Static Analysis for Ashtabula...\n" + "=" * 50)

    commands = [
        ("mypy (Type Checking)", "mypy ashtabula/"),
        ("ruff (Linting & Formatting)", "ruff check ashtabula/"),
        ("flake8 (General Linting)", "flake8 ashtabula/"),
        ("pylint (Code Quality Checks)", "pylint ashtabula/"),
        ("bandit (Security Scan)", "bandit -r ashtabula/"),
    ]

    for description, command in commands:
        run_command(description, command)

    print("\n✅ Static Analysis Completed.")

if __name__ == "__main__":
    run_static_analysis()