import pytest
import copy
import numpy as np
from numpy.testing import assert_almost_equal
from hothouse import blaster


def test_sun_blaster(location_champaign, altitude_champaign,
                     datetime_champaign, scene_soy):
    r"""Test creation & use of solar blaster."""
    nx = 512
    ny = 512
    date = datetime_champaign("sunrise")
    rb = scene_soy.get_sun_blaster(*location_champaign, date,
                                   altitude=altitude_champaign,
                                   nx=nx, ny=ny)
    assert_almost_equal(rb.solar_altitude, 7.8106451271435855)
    assert_almost_equal(rb.solar_distance, 694.869384765625)
    assert_almost_equal(rb.center, [574.76935, 254.03087, 532.0688],
                        decimal=5)
    rb.compute_distance(scene_soy)
    scene_soy.compute_solar_ppfd(*location_champaign, date,
                                 altitude=altitude_champaign)


class TestRayBlaster:
    r"""Tests for RayBlaster class."""

    cls = blaster.RayBlaster
    _instance_kws = dict(
        origins=np.array([
            [0.0, 0.0, 2.0],
            [0.5, 0.5, 2.0],
            [2.0, 2.0, 2.0],
        ], dtype="f4"),
        directions=np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ], dtype="f4"),
    )
    _expected_result = {
        'Ng': np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.6, 0.5],
            [0.0, 0.0, 0.0],
        ], "f4"),
        'geomID': np.array([0, 0, -1], "i4"),
        'primID': np.array([0, 4, -1], "i4"),
        'tfar': np.array([2.0, 0.4, 1e37], "f4"),
        'u': np.array([1.0, 1.0, 0.0], "f4"),
        'v': np.array([0.0, 0.0, 0.0], "f4"),
    }

    @pytest.fixture(scope="class")
    def instance_kws(self):
        return self._instance_kws

    @pytest.fixture(scope="class")
    def instance(self, instance_kws):
        return self.cls(**instance_kws)

    @pytest.fixture(scope="class")
    def instance_multibounce(self, instance_kws):
        return self.cls(multibounce=True, power_threshold=0.1,
                        **instance_kws)

    @pytest.fixture(scope="class")
    def expected_result(self):
        return self._expected_result

    @pytest.fixture(scope="class")
    def get_bounces_empty(self, instance):
        nray = instance.origins.shape[0]

        def _get_bounces_empty(nbounce):
            return {
                'nbounce': np.zeros((nray, ), "i4"),
                'Ng': np.zeros((nray, nbounce, 3), "f4"),
                'geomID': -1 * np.ones((nray, nbounce), "i4"),
                'primID': -1 * np.ones((nray, nbounce), "i4"),
                'tfar': 1e37 * np.ones((nray, nbounce), "f4"),
                'u': np.zeros((nray, nbounce), "f4"),
                'v': np.zeros((nray, nbounce), "f4"),
                'power': np.zeros((nray, nbounce), "f4"),
                'ray_dir': np.zeros((nray, nbounce, 3), "f4"),
            }

        return _get_bounces_empty

    @pytest.fixture(scope="class")
    def expected_bounces(self, get_bounces_empty):
        out = get_bounces_empty(6)
        idx_hit = (1, slice(1, 3))
        out['nbounce'][:] = [2, 6, 0]
        out['Ng'][idx_hit] = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.6, 0.5],
        ], "f4")
        out['geomID'][idx_hit] = 0
        out['primID'][idx_hit] = [0, 4]
        out['tfar'][idx_hit] = 1.5999
        out['u'][1, 2] = 1.0
        out['v'][1, 1] = 0.5
        out['power'] = np.array([
            [0.5, 0.25, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.25, 0.125, 0.0625, 0.0625, 0.03125],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype="f4")
        out['ray_dir'][0, 0, 2] = 1.0
        out['ray_dir'][0, 1, 2] = -1.0
        out['ray_dir'][1] = np.array([
            [1.1689008e-16, 0.56939501, -0.82206404],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [-1.1689008e-16, -0.56939501, 0.82206404],
            [0.0, 0.0, 1.0],
        ], dtype="f4")
        return out

    def test_compute_distance(self, instance, scene_pyramid,
                              expected_result):
        r"""Test calculation of travel distance to scene."""
        actual = instance.compute_distance(scene_pyramid)
        assert_almost_equal(actual, expected_result['tfar'])

    def test_compute_count(self, instance, scene_pyramid,
                           expected_result, assert_dicts_almost_equal):
        r"""Test calculation of intersections with scene."""
        actual = instance.compute_count(scene_pyramid)
        assert_dicts_almost_equal(actual, expected_result)

    def test_compute_count_multibounce(self, instance_multibounce,
                                       scene_pyramid, expected_result,
                                       expected_bounces,
                                       assert_dicts_almost_equal):
        r"""Test calculation of intersections with scene."""
        actual = instance_multibounce.compute_count(scene_pyramid)
        assert_dicts_almost_equal(actual, expected_result,
                                  ignore_keys=['bounces'])
        assert 'bounces' in actual
        assert_dicts_almost_equal(actual['bounces'], expected_bounces)

    # def compute_flux_density(self, instance, scene_pyramid):


class TestOrthographicRayBlaster(TestRayBlaster):
    r"""Tests for RayBlaster class."""

    cls = blaster.OrthographicRayBlaster
    _instance_kws = dict(
        # center=np.array([0.5, 0.5, 2.0], dtype="f4"),
        # forward=np.array([0.0, 0.0, -1.0], dtype="f4"),
        # up=np.array([0.0, 1.0, 0.0], dtype="f4"),
        width=1.0, height=1.0,
        nx=2, ny=2,
    )
    _expected_result = {
        'Ng': np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.6, 0.5],
            [1.6, 0.0, 0.5],
            [0.0, 1.6, 0.5],
        ], "f4"),
        'geomID': np.array([0, 0, 0, 0], "i4"),
        'primID': np.array([0, 4, 5, 4], "i4"),
        'tfar': np.array([2.0, 2.0, 2.0, 2.0], "f4"),
        'u': np.array([1.0, 0.0, 0.0, 0.0], "f4"),
        'v': np.array([0.0, 0.0, 0.0, 1.0], "f4"),
    }

    @pytest.fixture(scope="class")
    def instance_kws(self, scene_pyramid):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            forward=-scene_pyramid.up,
            center=(scene_pyramid.ground + 2.0 * scene_pyramid.up),
            up=scene_pyramid.north,
        )
        return out

    @pytest.fixture(scope="class")
    def expected_bounces(self, get_bounces_empty):
        out = get_bounces_empty(2)
        out['nbounce'][:] = [2, 2, 2, 2]
        out['power'][:, 0] = 0.5
        out['power'][:, 1] = 0.25
        out['ray_dir'][0, 0, 2] = 1.0
        out['ray_dir'][:, 1, 2] = -1.0
        out['ray_dir'][1, 0, :] = np.array([
            1.1689008e-16, 5.6939501e-01, -8.2206404e-01,
        ], "f4")
        out['ray_dir'][2, 0, :] = np.array([
            5.6939501e-01, -1.1689008e-16, -8.2206404e-01,
        ], "f4")
        out['ray_dir'][3, 0, :] = np.array([
            1.1689008e-16, 5.6939501e-01, -8.2206404e-01,
        ], "f4")
        return out


class TestProjectionRayBlaster(TestRayBlaster):
    r"""Tests for ProjectionRayBlaster class."""

    cls = blaster.ProjectionRayBlaster
    _instance_kws = dict(
        fov_width=14.250032697803595,
        fov_height=14.250032697803595,
        width=0.5, height=0.5,
        nx=2, ny=2,
    )
    _expected_result = dict(
        TestOrthographicRayBlaster._expected_result,
        tfar=2.0310097 * np.ones((4, ), "f4"),
    )

    @pytest.fixture(scope="class")
    def instance_kws(self, scene_pyramid):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            forward=-scene_pyramid.up,
            center=(scene_pyramid.ground + 2.0 * scene_pyramid.up),
            up=scene_pyramid.north,
        )
        return out

    @pytest.fixture(scope="class")
    def expected_bounces(self, get_bounces_empty):
        out = get_bounces_empty(2)
        out['nbounce'][:] = [2, 2, 2, 2]
        out['power'][:, 0] = 0.5
        out['power'][:, 1] = 0.25
        out['ray_dir'] = np.array([
            [[-0.12309149, -0.12309149, 0.9847319],
             [-0.12309149, -0.12309149, -0.9847319]],
            [[-0.12309149, 0.45951235, -0.87960035],
             [-0.12309149, 0.12309149, -0.9847319]],
            [[0.45951235, -0.12309149, -0.87960035],
             [0.12309149, -0.12309149, -0.9847319]],
            [[0.12309149, 0.45951235, -0.87960035],
             [0.12309149, 0.12309149, -0.9847319]],
        ], "f4")
        return out


class TestSunRayBlaster(TestOrthographicRayBlaster):
    r"""Tests for SunRayBlaster."""

    cls = blaster.SunRayBlaster
    _instance_kws = dict(
        up=np.array([0.0, 1.0, 0.0], dtype="f4"),
        width=1.0, height=1.0,
        nx=2, ny=2,
    )
    _expected_result = {
        'Ng': np.array([
            [3.8525363e-12, 1.4012985e-45, 0.0000000e+00],
            [3.8525363e-12, 1.4012985e-45, 0.0000000e+00],
            [3.8525363e-12, 1.4012985e-45, 0.0000000e+00],
            [1.6000000e+00, 0.0000000e+00, 5.0000000e-01],
        ], "f4"),
        'geomID': np.array([-1, -1, -1, 0], "i4"),
        'primID': np.array([-1, -1, -1, 5], "i4"),
        'tfar': np.array([
            1e37, 1e37, 1e37, 1.8205224
        ], "f4"),
        'u': np.array([0.0, 0.0, 0.0, 0.8984771], "f4"),
        'v': np.array([0.0, 0.0, 0.0, 0.09985479], "f4"),
    }

    @pytest.fixture(scope="class")
    def instance_kws(self, location_champaign, altitude_champaign,
                     datetime_champaign, scene_pyramid):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            latitude=location_champaign[0],
            longitude=location_champaign[1],
            altitude=altitude_champaign,
            date=datetime_champaign("noon"),
            ground=scene_pyramid.ground,
            zenith=(scene_pyramid.ground + 2.0 * scene_pyramid.up),
            north=scene_pyramid.north,
            scene_limits=scene_pyramid.limits,
        )
        return out

    @pytest.fixture(scope="class")
    def expected_bounces(self, get_bounces_empty):
        out = get_bounces_empty(6)
        out['nbounce'][:] = [0, 0, 0, 6]
        out['Ng'][3] = np.array([
            [0.0, 0.0, 0.0],
            [-0.0, -0.0, -1.0],
            [-0.0, 1.6, 0.5],
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ], "f4")
        out['geomID'][3] = [-1, 0, 0, -1, 0, -1]
        out['primID'][3] = [-1,  1,  4, -1,  5, -1]
        out['power'][3] = [0.5, 0.25, 0.125, 0.0625, 0.0625, 0.03125]
        out['ray_dir'][3] = np.array([
            [0.55538166, 0.287476, -0.78032607],
            [-0.01224552, 0.287476, -0.95770955],
            [-0.01224552, 0.287476, 0.95770955],
            [-0.01224552, 0.287476, -0.95770955],
            [-0.01224552, -0.7816387, 0.6236112],
            [-0.01224552, 0.287476, 0.95770955],
        ], "f4")
        out['tfar'][3] = [
            1e37, 1.6672263e-01, 6.1002281e-03, 1e37, 2.7471662e-01, 1e37
        ]
        out['u'][3] = [0.0, 0.05197027, 0.00371126, 0.0, 0.7279256, 0.0]
        out['v'][3] = [0.0, 0.94439167, 0.94609815, 0.0, 0.11082296, 0.0]
        return out


class TestSphericalRayBlaster(TestProjectionRayBlaster):

    cls = blaster.SphericalRayBlaster

    @pytest.fixture(scope="class")
    def expected_result(self, instance_kws):
        out = copy.deepcopy(self._expected_result)
        order = np.array([3, 2, 0, 1], "i4")
        for k in list(out.keys()):
            out[k] = out[k][order, ...]
        if not instance_kws.get('dont_include_center', False):
            out['tfar'] = np.hstack([
                np.array([2.4], "f4"),
                out['tfar'] + 2 / np.cos(
                    np.radians(instance_kws['fov_height']))
            ])
            out['Ng'] = np.vstack([
                np.array([0.0, 1.6, 0.5], "f4"),
                out['Ng']
            ])
            out['geomID'] = np.hstack([0, out['geomID']])
            out['primID'] = np.hstack([4, out['primID']])
            out['u'] = np.hstack([1.0, out['u']])
            out['v'] = np.hstack([0.0, out['v']])
        return out

    @pytest.fixture(scope="class")
    def expected_bounces(self, get_bounces_empty, instance_kws):
        if instance_kws.get('dont_include_center', False):
            nbounce = 2
            nray = 4
        else:
            nbounce = 6
            nray = 5
        out = get_bounces_empty(nbounce)
        out['nbounce'][:] = 2
        out['power'][:, 0] = 0.5
        out['power'][:, 1] = 0.25
        out['ray_dir'] = np.zeros((nray, nbounce, 3), "f4")
        out['ray_dir'][(nray - 4):, :2, :] = np.array([
            [[0.12309149, 0.45951235, -0.87960035],
             [0.12309149, 0.12309149, -0.9847319]],
            [[0.45951235, -0.12309149, -0.87960035],
             [0.12309149, -0.12309149, -0.9847319]],
            [[-0.12309149, -0.12309149, 0.9847319],
             [-0.12309149, -0.12309149, -0.9847319]],
            [[-0.12309149, 0.45951235, -0.87960035],
             [-0.12309149, 0.12309149, -0.9847319]],
        ], "f4")
        if not instance_kws.get('dont_include_center', False):
            out['nbounce'][0] = 6
            out['Ng'][0, 1:3, :] = np.array([
                [0.0, 0.0, -1.0],
                [0.0, 1.6, 0.5],
            ], "f4")
            out['geomID'][0, 1:3] = 0
            out['primID'][0, 1:3] = [0, 4]
            out['power'][0][2:] = [0.125, 0.0625, 0.0625, 0.03125]
            out['ray_dir'][0] = np.array([
                [0.0, 0.56939501, -0.82206404],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, -0.56939501, 0.82206404],
                [0.0, 0.0, 1.0],
            ], "f4")
            out['tfar'][0, 1:3] = [1.5998999, 1.5998999]
            out['u'][0, 1:3] = [0.0, 1.0]
            out['v'][0, 1:3] = [0.5, 0.0]
        return out

    @pytest.fixture(scope="class")
    def instance_kws(self, scene_pyramid):
        out = copy.deepcopy(self._instance_kws)
        out.pop('width')
        out.pop('height')
        out.update(
            forward=-scene_pyramid.up,
            center=(scene_pyramid.ground + 4.0 * scene_pyramid.up),
            up=scene_pyramid.north,
            fov_width=np.float32(270),
            fov_height=np.float32(np.degrees(0.17496904566568888)),
            nx=4, ny=1,
        )
        return out
