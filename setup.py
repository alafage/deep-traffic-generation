#! /usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="deep_traffic_generation",
    version="0.0.0",
    description="Description",
    author="Adrien Lafage",
    author_email="",
    url="https://github.com/alafage",
    install_requires=[
        "pytorch-lightning",
        "traffic",
        "numba",
        "sphinx",
        "sphinx_rtd_theme",
        "sphinx-copybutton",
        "pyvinecopulib",
    ],
    packages=find_packages(),
)
