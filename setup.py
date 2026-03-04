from setuptools import setup, find_packages

setup(
    name="wheat-spike-analyzer",
    version="1.1.0",
    author="Nicolas SALAS",
    description="Analyse phénotypique d'épis de blé par YOLO OBB/Seg",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NicoSalas34/wheat-spike-analyzer",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.3.0",
        "numpy>=1.26.0",
        "opencv-python-headless>=4.10.0",
        "scikit-image>=0.22.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0",
        "Pillow>=10.0.0",
        "albumentations>=2.0.0",
        "tqdm>=4.66.0",
        "psutil>=5.9.0",
        "Flask>=3.0.0",
    ],
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'wheat-analyzer=src.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
