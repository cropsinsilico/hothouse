#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
import versioneer
import sys
import os
from pkg_resources import resource_filename
from sys import platform as _platform

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
    "Cython>=0.29.20",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]


# These routines are taken from yt
def in_conda_env():
    return any(s in sys.version for s in ("Anaconda", "Continuum", "conda"))


def check_for_pyembree():
    try:
        fn = resource_filename("pyembree", "rtcore.pxd")
    except ImportError:
        return None
    return os.path.dirname(fn)


std_libs = []


def append_embree_info(exts):
    embree_prefix = os.path.abspath(check_for_pyembree())
    embree_inc_dir = [os.path.join(embree_prefix, "include")]
    embree_lib_dir = [os.path.join(embree_prefix, "lib")]
    if in_conda_env():
        conda_basedir = os.path.dirname(os.path.dirname(sys.executable))
        embree_inc_dir.append(os.path.join(conda_basedir, "include"))
        embree_lib_dir.append(os.path.join(conda_basedir, "lib"))

    if _platform == "darwin":
        embree_lib_name = "embree.2"
    else:
        embree_lib_name = "embree"

    for ext in exts:
        ext.include_dirs += embree_inc_dir + [numpy.get_include()]
        ext.library_dirs += embree_lib_dir
        ext.language = "c++"
        ext.libraries += std_libs
        ext.libraries += [embree_lib_name]

    return exts


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
    ext_modules=append_embree_info(cythonize("**/*.pyx")),
)
