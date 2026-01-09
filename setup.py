from setuptools import setup, find_packages

setup(
    name="wheat-spike-analyzer",
    version="1.0.0",
    author="Nicolas SALAS",
    description="Analyse phénotypique d'épis de blé",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "opencv-python>=4.9.0",
        "scikit-image>=0.22.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.0",
        "matplotlib>=3.8.0",
    ],
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'wheat-analyzer=src.main:main',
        ],
    },
)
