"""
Setup script for ByteTrack Light Post Tracking System
"""

from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Try to read requirements from multiple locations
requirements = []
possible_req_files = [
    "requirements.txt",
    "argus_track/requirements.txt"
]

for req_file in possible_req_files:
    if os.path.exists(req_file):
        with open(req_file, "r", encoding="utf-8") as fh:
            requirements = [
                line.strip() 
                for line in fh 
                if line.strip() and not line.startswith("#")
            ]
        break

# If no requirements file found, use minimal requirements
if not requirements:
    requirements = [
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.4.0",
        "pandas>=1.3.0",
        "Pillow>=8.0.0",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.6.0",
        "python-dateutil>=2.8.0",
        "psutil>=5.8.0"
    ]

setup(
    name="argus-track",
    version="1.0.0",
    author="Light Post Tracking Team",
    author_email="joaquin.olivera@gmail.com",
    description="ByteTrack implementation optimized for light post tracking with GPS integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bell-South/ArgusTrack.git",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "argus_track=argus_track.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "argus_track": ["config/*.yaml", "config/*.json"],
    },
)