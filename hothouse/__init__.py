# -*- coding: utf-8 -*-

"""Top-level package for hothouse."""

__author__ = """Matthew Turk"""
__email__ = "matthewturk@gmail.com"

try:
    from ._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover
    __version__ = "unknown version"
    __version_tuple__ = (0, 0, "unknown version")

from .model import Model
from .blaster import OrthographicRayBlaster, SunRayBlaster
from .scene import Scene


__all__ = ["Scene", "Model", "OrthographicRayBlaster", "SunRayBlaster"]
