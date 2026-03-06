#!/usr/bin/env python3
"""
Setup script for Agno GraphRAG
Makes the package installable with pip
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else "Agno-based GraphRAG system for voxel knowledge bases"

setup(
    name="agno_graphrag",
    version="1.0.0",
    author="GraphRAG Team",
    description="Agno-based GraphRAG system for voxel knowledge bases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "agno>=0.1.0",
        "neo4j==5.15.0",
        "openai>=1.6.1,<2.0.0",
        "python-dotenv==1.0.0",
        "pydantic>=2.5.0,<2.10.0",
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "trimesh>=3.20.0",
    ],
    entry_points={
        "console_scripts": [
            "agno-graphrag=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

