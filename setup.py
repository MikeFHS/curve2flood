"""
Setup script for the Curve2Flood package.
"""

from setuptools import setup, find_packages

setup(
    name="Curve2Flood",
    version="0.3.1",
    description="Flood mapping tool based on DEM and other inputs.",
    author="Michael Follum",
    author_email="mike@follumhydro.com",
    url="https://github.com/MikeFHS/curve2flood",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "gdal",
        "fastparquet",
        "rasterio",
        "geopandas",
        "pyyaml",
        "shapely",
        "scipy",
        "numba",
        "tqdm",
        "pyproj",
    ],
    entry_points={
        "console_scripts": [
            "curve2flood=curve2flood.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    license="GPL-3.0",
)
