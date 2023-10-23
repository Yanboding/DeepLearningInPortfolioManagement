from setuptools import setup, find_packages

setup(
    name="portfolio-management-package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.11",
        "pandas>=1.3",
        "Pillow>=10",
        "torch>=2.0",
        "xarray>=0.20",
        "bottleneck>=1.3.5",
        "matplotlib>=3.3.4",
        "ccxt>=4.0",
        "tqdm>=4.65",
        "gymnasium>=0.28",
        "gpytorch"
    ],
)