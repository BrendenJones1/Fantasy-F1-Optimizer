#!/usr/bin/env python3
"""
Simple setup script for Fantasy F1 Optimizer
"""

from setuptools import setup, find_packages

setup(
    name="fantasy-f1-optimizer",
    version="1.0.0",
    description="Machine learning system for Fantasy Formula 1 optimization",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.5.0", 
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mysql-connector-python>=8.0.0",
        "requests>=2.31.0",
        "joblib>=1.3.0",
    ],
    python_requires=">=3.8",
) 