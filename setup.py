#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="napari-slice-propagator",
    use_scm_version=False,
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Semi-automated 3D segmentation tool with slice propagation and active contours",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "napari[all]",
        "numpy",
        "scikit-image",
        "scipy",
        "qtpy",
    ],
    entry_points={
        "napari.manifest": [
            "napari-slice-propagator = napari_slice_propagator:napari.yaml",
        ],
    },
    include_package_data=True,
    package_data={
        "napari_slice_propagator": ["napari.yaml"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: napari",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)