#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="crypto_trader",
    version="0.1.0",
    description="Multi-Agent AI Trading System with Famous Trader Personas",
    author="CryptoTrader Team",
    author_email="info@cryptotrader.example",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "flask>=2.3.0",
        "flask-socketio>=5.3.0",
        "ccxt>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "dash>=2.13.0",
        "dash-bootstrap-components>=1.4.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "crypto-trader=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
) 