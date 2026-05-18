import os
import pytest
import pytz
import datetime
import numpy as np


class NestedAssertionError(AssertionError):

    def __init__(self, nested):
        self.nested = nested
        msg = ''
        for k, v in nested.items():
            msg += f'\n\n{k}\n\t' + v.replace('\n', '\n\t')
        super(NestedAssertionError, self).__init__(msg)


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
        elif name == 'midnight':
            args = (2020, 6, 17, 0, 25, 0, 0)
        return datetime.datetime(*args, tz_champaign)

    return _datetime_champaign


@pytest.fixture(scope="session")
def datadir():
    r"""Directory containing test data."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


@pytest.fixture(scope="session")
def fname_ply_soy():
    r"""Path to Ply file containing a simulated soy plant."""
    from hothouse.datasets import PLANTS
    return PLANTS.fetch("fullSoy_2-12a.ply")


@pytest.fixture(scope="session")
def fname_obj_sphere(datadir):
    r"""Path to ObjWavefront file containing a sphere."""
    return os.path.join(datadir, 'sphere.obj')


@pytest.fixture(scope="session")
def fname_ply_pyramid(datadir):
    r"""Path to Ply file containing a pyramid."""
    return os.path.join(datadir, 'pyramid.ply')


@pytest.fixture(scope="session")
def fname_obj_pyramid(datadir):
    r"""Path to ObjWavefront file containing a pyramid."""
    return os.path.join(datadir, 'pyramid.obj')


@pytest.fixture(scope="session")
def geometry_fname(datadir):
    r"""Factory for test geometry file names.

    Args:
        name (str): Geometry name.
        ftype (str, optional): Geometry file type.

    Returns:
        str: Path to the file containing the described test geometry.

    """

    def _geometry_fname(name, ftype=None):
        if ftype is None:
            ftype = 'obj' if name in ['sphere'] else 'ply'
        if name == 'soy' and ftype == 'ply':
            from hothouse.datasets import PLANTS
            return PLANTS.fetch("fullSoy_2-12a.ply")
        fname = os.path.join(datadir, f'{name}.{ftype}')
        if not os.path.isfile(fname):
            raise NotImplementedError(f"No test data for \"{name}\" "
                                      f"(ftype = \"{ftype}\"")
        return fname

    return _geometry_fname


@pytest.fixture(scope="session")
def geometry_model(geometry_fname):
    r"""Cached factory for test models.

    Args:
        name (str): Geometry name.
        ftype (str, optional): Geometry file type.

    Returns:
        hothouse.model.Model: Test model containing the described
            geometry.

    """
    from hothouse.model import Model
    cache = {}

    def _geometry_model(name, ftype=None, **kwargs):
        key = (name, ftype)
        if key in cache:
            return cache[key]
        defaults = {}
        if name != 'soy':
            defaults.update(
                attributes=dict(
                    reflectance=0.5,
                    transmittance=0.25,
                )
            )
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        fname = geometry_fname(name, ftype=ftype)
        out = Model.from_file(fname, **kwargs)
        cache[key] = out
        return out

    return _geometry_model


@pytest.fixture(scope="session")
def geometry_scene(geometry_model):
    r"""Cached factory for test scenes.

    Args:
        name (str): Geometry name.
        ftype (str, optional): Geometry file type.

    Returns:
        hothouse.scene.Scene: Test scene containing the described
            geometry.

    """
    from hothouse.scene import Scene
    cache = {}

    def _geometry_scene(name, ftype=None, **kwargs):
        key = (name, ftype)
        if key in cache:
            return cache[key]
        model = geometry_model(name, ftype=ftype)
        defaults = {}
        if name == "soy":
            defaults.update(
                ground=np.array([0.0, 0.0, 200.0], dtype="f8"),
                up=np.array([0.0, 0.0, 1.0], dtype="f8"),
                north=np.array([0.0, 1.0, 0.0], dtype="f8"),
            )
        elif name == "pyramid":
            defaults.update(
                ground=np.array([0.5, 0.5, 0.0], "f8"),
                up=np.array([0.0, 0.0, 1.0], dtype="f8"),
                north=np.array(
                    [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
                    dtype="f8"),
            )
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        s = Scene(**kwargs)
        s.add_component(model)
        cache[key] = s
        return s

    return _geometry_scene


@pytest.fixture(scope="session")
def tolerances_bounces():
    return {'rtol': 1e-6}


@pytest.fixture(scope="session")
def tolerances_solar():
    return {'rtol': 1e-6}


@pytest.fixture(scope="session")
def assert_almost_equal():
    r"""Assert that arrays are close."""

    def _assert_almost_equal(a, b, decimals=7, **kwargs):
        np.testing.assert_almost_equal(a, b, decimals=decimals, **kwargs)

    return _assert_almost_equal


@pytest.fixture(scope="session")
def assert_dicts_almost_equal(assert_almost_equal):
    r"""Assert that dictionaries of numpy arrays are almost equal."""

    def _assert_dicts_almost_equal(a, b, ignore_keys=None,
                                   only_keys=None, **kwargs):
        a_keys = list(sorted(a.keys()))
        b_keys = list(sorted(b.keys()))
        if ignore_keys:
            a_keys = [k for k in a_keys if k not in ignore_keys]
            b_keys = [k for k in b_keys if k not in ignore_keys]
        if only_keys:
            a_keys = [k for k in a_keys if k in only_keys]
            b_keys = [k for k in b_keys if k in only_keys]
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
def assert_allclose():
    r"""Assert that arrays are close."""

    def _assert_allclose(a, b, rtol=1e-07, atol=1e-15, **kwargs):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, **kwargs)

    return _assert_allclose


@pytest.fixture(scope="session")
def assert_nested_allclose(assert_allclose):
    from collections import OrderedDict

    def _assert_nested_allclose(a, b, ignore_keys=None,
                                only_keys=None, **kwargs):
        errors = {}
        if isinstance(b, (list, tuple)):
            assert isinstance(a, type(b))
            assert len(a) == len(b)
            for i, (ia, ib) in enumerate(zip(a, b)):
                try:
                    _assert_nested_allclose(ia, ib, **kwargs)
                except AssertionError as e:
                    if isinstance(e, NestedAssertionError):
                        for kerr, verr in e.nested.items():
                            errors[f'{i}->{kerr}'] = verr
                    else:
                        errors[f'{i}'] = e.args[0]
        elif isinstance(b, (dict, OrderedDict)):
            assert isinstance(a, type(b))
            a_keys = list(sorted(a.keys()))
            b_keys = list(sorted(b.keys()))
            if ignore_keys:
                a_keys = [k for k in a_keys if k not in ignore_keys]
                b_keys = [k for k in b_keys if k not in ignore_keys]
            if only_keys:
                a_keys = [k for k in a_keys if k in only_keys]
                b_keys = [k for k in b_keys if k in only_keys]
            assert a_keys == b_keys
            for k in b_keys:
                try:
                    _assert_nested_allclose(a[k], b[k], **kwargs)
                except AssertionError as e:
                    if isinstance(e, NestedAssertionError):
                        for kerr, verr in e.nested.items():
                            errors[f'{k}->{kerr}'] = verr
                    else:
                        errors[k] = e.args[0]
        elif isinstance(b, (np.ndarray, float)):
            assert_allclose(a, b, **kwargs)
        else:
            assert a == b
        if errors:
            raise NestedAssertionError(errors)

    return _assert_nested_allclose


@pytest.fixture(scope="session")
def assert_dicts_allclose(assert_allclose):
    r"""Assert that dictionaries of numpy arrays are almost equal."""

    def _assert_dicts_allclose(a, b, ignore_keys=None, **kwargs):
        a_keys = list(sorted(a.keys()))
        b_keys = list(sorted(b.keys()))
        if ignore_keys:
            a_keys = [k for k in a_keys if k not in ignore_keys]
            b_keys = [k for k in b_keys if k not in ignore_keys]
        assert a_keys == b_keys
        for k in b_keys:
            try:
                assert_allclose(a[k], b[k], **kwargs)
            except AssertionError:
                print(k)
                if a[k].shape == b[k].shape:
                    print(a[k] == b[k])
                print(a[k])
                raise

    return _assert_dicts_allclose


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
