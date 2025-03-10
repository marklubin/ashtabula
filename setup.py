#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="ashtabula",
    version="0.1.0",
    description="Ashtabula AI framework",
    author="Mark Lubin",
    author_email="mark@niteshift.ai",
    url="https://github.com/niteshift-ai/ashtabula",
    packages=find_packages(exclude=["tests", "test_*", "*.tests", "*.tests.*", "tests.*"]),
    include_package_data=True,
    python_requires=">=3.8.1",
    install_requires=[
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "accelerate>=0.25.0",
        "numpy>=1.24.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.23.0",
        "mypy>=1.7.0",
        "ruff>=0.1.9",
        "flake8>=6.1.0",
        "pylint>=3.0.0",
        "bandit>=1.7.0",
        "huggingface_hub>=0.20.0",
        "websockets>=13.1",
        "transitions>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "ashtabula=ashtabula.main:run",
            "ashtabula-download-models=scripts.download_models:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)