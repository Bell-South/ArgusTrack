
"""
Setup script for Argus Track with Ultralytics ByteTrack integration
"""

from setuptools import setup, find_packages

setup(
    name="argus-track",
    version="1.0.0",
    description="Enhanced Light Post Tracking with Ultralytics ByteTrack",
    author="Argus Track Team",
    author_email="joaquin.olivera@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "opencv-python>=4.5.0",
        "filterpy>=1.4.5",
        "pyproj>=3.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "geojson>=2.5.0",
    ],
    entry_points={
        "console_scripts": [
            "argus-track-ultralytics=ultralytics_main:main",
        ],
    },
)