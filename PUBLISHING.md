# Publishing MCP Manager to PyPI

This guide explains how to publish MCP Manager to PyPI so users can run it with `uvx mcp-manager`.

## Package Built Successfully ✅

The package has been built and is ready for publishing:
- `dist/mcp_manager-0.1.0-py3-none-any.whl` (wheel distribution)
- `dist/mcp_manager-0.1.0.tar.gz` (source distribution)

## Prerequisites

1. **PyPI Account**: Create accounts at:
   - TestPyPI (for testing): https://test.pypi.org/account/register/
   - PyPI (for production): https://pypi.org/account/register/

2. **API Tokens**: Generate API tokens for authentication:
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. **Install twine** (if not already installed):
   ```bash
   uv pip install twine
   ```

## Step 1: Test on TestPyPI (Recommended)

Test your package on TestPyPI first to ensure everything works:

```bash
# Upload to TestPyPI
uv run twine upload --repository testpypi dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your TestPyPI token starting with pypi->
```

Then test installation from TestPyPI:

```bash
# Test with uvx
uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mcp-manager

# Or test with pip
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mcp-manager
```

**Note**: The `--extra-index-url https://pypi.org/simple` is needed because TestPyPI doesn't host the dependencies (textual, pydantic, etc.).

## Step 2: Publish to PyPI (Production)

Once you've verified everything works on TestPyPI:

```bash
# Upload to PyPI
uv run twine upload dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your PyPI token starting with pypi->
```

## Step 3: Verify Installation

After publishing to PyPI, users can install and run your package:

```bash
# Run directly with uvx (no installation needed)
uvx mcp-manager

# Or install globally
pip install mcp-manager

# Or install with uv
uv pip install mcp-manager

# Then run
mcp-manager
```

## Alternative: Using .pypirc for Authentication

To avoid typing credentials each time, create a `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your PyPI token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your TestPyPI token>
```

Then you can upload without prompts:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Troubleshooting

### Package Name Already Taken

If `mcp-manager` is already taken on PyPI, you'll need to:

1. Choose a different name (e.g., `mcp-server-manager`, `mcp-manager-tui`)
2. Update the name in `pyproject.toml`:
   ```toml
   name = "mcp-server-manager"  # or your chosen name
   ```
3. Rebuild the package:
   ```bash
   rm -rf dist/
   uv run python -m build
   ```
4. Upload with the new name

### Version Already Exists

If you need to re-upload after fixing an issue:

1. Increment the version in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # or next appropriate version
   ```
2. Rebuild:
   ```bash
   rm -rf dist/
   uv run python -m build
   ```
3. Upload the new version

### Testing Changes Without Publishing

To test changes locally before publishing:

```bash
# Install in editable mode
uv pip install -e .

# Test the CLI
mcp-manager

# Or run directly
uv run python -m mcp_manager
```

## Updating the Package

When you make changes and want to release a new version:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.2.0"  # follow semantic versioning
   ```

2. **Update CHANGELOG** (create one if needed):
   ```markdown
   ## [0.2.0] - 2025-11-06
   ### Added
   - New feature X
   ### Fixed
   - Bug Y
   ```

3. **Rebuild**:
   ```bash
   rm -rf dist/
   uv run python -m build
   ```

4. **Test on TestPyPI** (optional but recommended):
   ```bash
   twine upload --repository testpypi dist/*
   ```

5. **Publish to PyPI**:
   ```bash
   twine upload dist/*
   ```

6. **Tag the release** in git:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

## Quick Reference

```bash
# Build package
uv run python -m build

# Test on TestPyPI
uv run twine upload --repository testpypi dist/*
uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mcp-manager

# Publish to PyPI
uv run twine upload dist/*

# Verify
uvx mcp-manager
```

## Package Information

- **Package Name**: `mcp-manager`
- **Version**: `0.1.0`
- **PyPI URL** (after publishing): https://pypi.org/project/mcp-manager/
- **Command**: `mcp-manager` or `uvx mcp-manager`

## Support

If you encounter issues:
1. Check the error messages carefully
2. Verify your API token is valid
3. Ensure the package name is available
4. Check that all metadata in `pyproject.toml` is correct
5. Try TestPyPI first to catch issues early

---

**Ready to publish!** The package is built and tested. Follow Step 1 above to publish to TestPyPI first, then Step 2 to publish to production PyPI.
