#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup
from Cython.Build import cythonize
import numpy
import sys
import os
from pkg_resources import resource_filename


# These routines are taken from yt
def in_conda_env():
    return any(s in sys.version for s in ("Anaconda", "Continuum", "conda"))


def check_for_pyembree():
    try:
        fn = resource_filename("pyembree", "rtcore.pxd")
    except ImportError:
        return None
    return os.path.dirname(fn)


def append_embree_info(exts):
    embree_prefix = os.path.abspath(check_for_pyembree())
    embree_inc_dir = [os.path.join(embree_prefix, "include")]
    embree_lib_dir = [os.path.join(embree_prefix, "lib")]
    if in_conda_env():
        conda_basedir = os.environ['CONDA_PREFIX']
        embree_inc_dir.append(os.path.join(conda_basedir, "include"))
        embree_lib_dir.append(os.path.join(conda_basedir, "lib"))

    embree_lib_name = "embree4"

    for ext in exts:
        ext.include_dirs += embree_inc_dir + [numpy.get_include()]
        ext.library_dirs += embree_lib_dir
        ext.language = "c++"
        ext.libraries += [embree_lib_name]

    return exts


setup(
    ext_modules=append_embree_info(cythonize("**/*.pyx")),
)
