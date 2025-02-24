#!/usr/bin/env python3
"""
Setup script for the K-ISA Python package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kisa",
    version="0.1.0",
    author="K-ISA Team",
    author_email="info@kisa-project.org",
    description="Python implementation of K-ISA (Knowledge-based Instruction Set Architecture)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kisa",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
        ],
    },
) 