import os
import pytest
import pytz
import datetime
import numpy as np
from numpy.testing import assert_almost_equal


@pytest.fixture(scope="session")
def tz_champaign():
    r"""Timezone for Champaign, IL"""
    return pytz.timezone("America/Chicago")


@pytest.fixture(scope="session")
def location_champaign():
    r"""Latitude & longitude of Champaign, IL (in degrees)."""
    return (40.1164, -88.2434)


@pytest.fixture(scope="session")
def altitude_champaign():
    r"""Altitude of Champaign, IL (in meters)."""
    return 224.0


@pytest.fixture(scope="session")
def datetime_champaign(tz_champaign):
    r"""Factory for getting a named datetime in Champaign IL."""

    def _datetime_champaign(name):
        if name == 'noon':
            args = (2020, 6, 17, 12, 0, 0, 0)
        elif name == 'sunrise':
            args = (2020, 6, 17, 5, 23, 0, 0)
        elif name == 'sunset':
            args = (2020, 6, 17, 19, 25, 0, 0)
        return datetime.datetime(*args, tz_champaign)

    return _datetime_champaign


@pytest.fixture(scope="session")
def datadir():
    r"""Directory containing test data."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


@pytest.fixture(scope="session")
def model_soy():
    r"""Model containing test soy data."""
    from hothouse.datasets import PLANTS
    from hothouse.model import Model
    fname = PLANTS.fetch("fullSoy_2-12a.ply")
    return Model.from_ply(fname)


@pytest.fixture(scope="session")
def model_sphere(datadir):
    r"""Model containg test sphere data."""
    from hothouse.model import Model
    fname = os.path.join(datadir, 'sphere.obj')
    return Model.from_obj(fname, reflectance=0.5, transmittance=0.25)


@pytest.fixture(scope="session")
def model_pyramid(datadir):
    r"""Model containg test pyramid data."""
    from hothouse.model import Model
    fname = os.path.join(datadir, 'pyramid.ply')
    return Model.from_ply(fname, reflectance=0.5, transmittance=0.25)


@pytest.fixture(scope="session")
def scene_soy(model_soy):
    r"""Scene containing test soy data."""
    from hothouse.scene import Scene
    s = Scene(
        ground=np.array([0.0, 0.0, 200.0], dtype="f4"),
        up=np.array([0.0, 0.0, 1.0], dtype="f4"),
        north=np.array([0.0, 1.0, 0.0], dtype="f4"),
    )
    s.add_component(model_soy)
    return s


@pytest.fixture(scope="session")
def scene_sphere(model_sphere):
    r"""Scene containing test sphere data."""
    from hothouse.scene import Scene
    s = Scene()
    s.add_component(model_sphere)
    return s


@pytest.fixture(scope="session")
def scene_pyramid(model_pyramid):
    r"""Scene containing test pyramid data."""
    from hothouse.scene import Scene
    s = Scene(
        ground=np.array([0.5, 0.5, 0.0], "f4"),
        up=np.array([0.0, 0.0, 1.0], dtype="f4"),
        north=np.array(
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0], dtype="f4"),
    )
    s.add_component(model_pyramid)
    return s


@pytest.fixture(scope="session")
def assert_dicts_almost_equal():
    r"""Assert that dictionaries of numpy arrays are almost equal."""

    def _assert_dicts_almost_equal(a, b, ignore_keys=None, **kwargs):
        a_keys = list(sorted(a.keys()))
        b_keys = list(sorted(b.keys()))
        if ignore_keys:
            a_keys = [k for k in a_keys if k not in ignore_keys]
            b_keys = [k for k in b_keys if k not in ignore_keys]
        assert a_keys == b_keys
        for k in b_keys:
            try:
                assert_almost_equal(a[k], b[k], **kwargs)
            except AssertionError:
                print(k)
                if a[k].shape == b[k].shape:
                    print(a[k] == b[k])
                print(a[k])
                raise

    return _assert_dicts_almost_equal


@pytest.fixture(scope="session")
def uvec_x():
    r"""Unit x vector."""
    return np.array([1.0, 0.0, 0.0])


@pytest.fixture(scope="session")
def uvec_y():
    r"""Unit y vector."""
    return np.array([0.0, 1.0, 0.0])


@pytest.fixture(scope="session")
def uvec_z():
    r"""Unit z vector."""
    return np.array([0.0, 0.0, 1.0])


@pytest.fixture
def uvec(uvec_x, uvec_y, uvec_z):
    r"""Iterate over the coordinate unit vectors."""
    for u in [uvec_x, uvec_y, uvec_z]:
        yield u
