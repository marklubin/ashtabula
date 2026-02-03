#!/bin/bash
# Automated publishing script for MCP Manager

set -e  # Exit on error

echo "🚀 MCP Manager Publishing Script"
echo "================================"
echo ""

# Check if dist exists and is not empty
if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
    echo "⚠️  Found existing dist/ directory"
    read -p "Do you want to clean it and rebuild? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🧹 Cleaning dist/"
        rm -rf dist/
    fi
fi

# Build if needed
if [ ! -d "dist" ] || [ ! "$(ls -A dist)" ]; then
    echo "📦 Building package..."
    uv run python -m build
    echo "✅ Package built successfully"
    echo ""
fi

# Ask where to publish
echo "Where do you want to publish?"
echo "1) TestPyPI (recommended for first time)"
echo "2) PyPI (production)"
echo "3) Both (TestPyPI first, then PyPI)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "📤 Publishing to TestPyPI..."
        echo "Username: __token__"
        echo "Password: <paste your TestPyPI token>"
        uv run twine upload --repository testpypi dist/*
        echo ""
        echo "✅ Published to TestPyPI!"
        echo ""
        echo "Test it with:"
        echo "  uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mcp-manager"
        ;;
    2)
        echo ""
        echo "📤 Publishing to PyPI..."
        echo "Username: __token__"
        echo "Password: <paste your PyPI token>"
        uv run twine upload dist/*
        echo ""
        echo "✅ Published to PyPI!"
        echo ""
        echo "Test it with:"
        echo "  uvx mcp-manager"
        ;;
    3)
        echo ""
        echo "📤 Publishing to TestPyPI first..."
        echo "Username: __token__"
        echo "Password: <paste your TestPyPI token>"
        uv run twine upload --repository testpypi dist/*
        echo ""
        echo "✅ Published to TestPyPI!"
        echo ""
        echo "Test it with:"
        echo "  uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mcp-manager"
        echo ""
        read -p "Does it work? Ready to publish to PyPI? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            echo "📤 Publishing to PyPI..."
            echo "Username: __token__"
            echo "Password: <paste your PyPI token>"
            uv run twine upload dist/*
            echo ""
            echo "✅ Published to PyPI!"
            echo ""
            echo "Test it with:"
            echo "  uvx mcp-manager"
        else
            echo "❌ Cancelled PyPI publishing"
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Done!"
