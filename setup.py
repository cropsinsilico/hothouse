#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup
from Cython.Build import cythonize
import numpy
import sys
import os
import importlib


# These routines are taken from yt
def in_conda_env():
    return any(s in sys.version for s in ("Anaconda", "Continuum", "conda"))


class EmbreePaths(object):

    def __init__(self, verbose=False):
        self.version = importlib.metadata.version("embreex")
        self.lib_name = "embree" + self.version.split('.')[0]
        self.libs = [self.lib_name]
        fn = importlib.resources.files("embreex") / "rtcore.pxd"
        self.prefix = os.path.abspath(os.path.dirname(fn))
        self.inc_dir = [os.path.join(self.prefix, "include")]
        self.lib_dir = [os.path.join(self.prefix, "lib")]
        if in_conda_env():
            conda_basedir = os.environ['CONDA_PREFIX']
            self.inc_dir.append(os.path.join(conda_basedir, "include"))
            self.lib_dir.append(os.path.join(conda_basedir, "lib"))
        self.inc_dir += [numpy.get_include()]
        if verbose:
            print("PREFIX", self.prefix)
            print("VERSION", importlib.metadata.version("embreex"))
            print("INCLUDE_DIR", self.inc_dir)
            print("LIBRARY_DIR", self.lib_dir)
            print("LIBRARIES", self.libs)


embree_paths = EmbreePaths()


def append_embree_info(exts):
    for ext in exts:
        ext.include_dirs += embree_paths.inc_dir
        ext.library_dirs += embree_paths.lib_dir
        ext.language = "c++"
        ext.libraries += embree_paths.libs

    return exts

ext_modules = append_embree_info(cythonize(
    'hothouse/*.pyx',
    include_path=embree_paths.inc_dir,
    aliases={'EMBREE_INCLUDE_DIR': embree_paths.prefix},
    compiler_directives={'language_level': 2},
))

setup(
    ext_modules=ext_modules,
)
