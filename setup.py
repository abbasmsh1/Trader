#!/usr/bin/env python
"""
Crypto Trader setup script.
"""

from setuptools import setup, find_packages

setup(
    name="crypto_trader",
    version="0.1.0",
    description="Agent-based cryptocurrency trading system",
    author="Crypto Trader Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "python-dotenv",
        "requests",
        "ccxt",
        "scikit-learn",
        "matplotlib",
        "joblib",
        "Flask",
        "Flask-SocketIO",
    ],
    entry_points={
        "console_scripts": [
            "crypto-trader=main:run_main",
        ],
    },
) 