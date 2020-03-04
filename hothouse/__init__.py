# -*- coding: utf-8 -*-

"""Top-level package for hothouse."""

__author__ = """Matthew Turk"""
__email__ = "matthewturk@gmail.com"
__version__ = "0.1.0"

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .plant_model import PlantModel
from .blaster import OrthographicRayBlaster
from .scene import Scene
