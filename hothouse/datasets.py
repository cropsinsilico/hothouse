import pooch

# Going to set this up a bit later...
from . import __version__

_registry = {
    "fullSoy_2-12a.ply": "e12f192188058851289f0531dc456c6df31b562405b77e382e0f9e4b1c899108"
}

PLANTS = pooch.create(
    path=pooch.os_cache("hothouse"),
    base_url="https://github.com/MatthewTurk/hothouse/raw/{version}/data/",
    version=__version__,
    version_dev="master",
    env="HOTHOUSE_DATA_DIR",
    registry=_registry,
)
