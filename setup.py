#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=6.0",
    "pyembree>=0.1.6",
    "yggdrasil-framework>=0.8.7",
    "pooch>=0.3.1",
    "plyfile>=0.7",
    "numpy>=1.13.0",
    "traitlets>=4.3.3",
    "traittypes>=0.2.1",
    "pvlib>=0.7.2",
    "tables>=3.6.1",
    "pythreejs>=2.2.0",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]

setup(
    author="Matthew Turk",
    author_email="matthewturk@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Embree-based ray tracer for photosynthetic yields in plant canopies",
    entry_points={"console_scripts": ["hothouse=hothouse.cli:main",],},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="hothouse",
    name="hothouse",
    packages=find_packages(include=["hothouse"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/matthewturk/hothouse",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
