import pytest
import copy
import numpy as np
from hothouse import blaster, sun_calc


def test_sun_blaster(location_champaign, altitude_champaign,
                     datetime_champaign, scene_soy,
                     assert_allclose):
    r"""Test creation & use of solar blaster."""
    nx = 512
    ny = 512
    date = datetime_champaign("sunrise")
    rb = scene_soy.get_sun_blaster(*location_champaign, date,
                                   altitude=altitude_champaign,
                                   nx=nx, ny=ny)
    assert_allclose(rb.solar_altitude, 7.807668468792781)
    assert_allclose(rb.solar_distance, 694.869384765625)
    assert_allclose(
        rb.center,
        np.array([574.7942504882812, 254.042, 532.0391845703125], "f4"))
    rb.compute_distance(scene_soy)
    scene_soy.compute_solar_ppfd(*location_champaign, date,
                                 altitude=altitude_champaign)


class TestRayBlaster:
    r"""Tests for RayBlaster class."""

    cls = blaster.RayBlaster
    _reorder_rays = None
    _instance_kws = dict(
        origins=np.array([
            [0.25, 0.5, 2.0],
            [0.5, 0.75, 2.0],
            [0.5, 0.25, 2.0],
            [0.75, 0.5, 2.0],
        ], dtype="f4"),
        directions=np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ], dtype="f4"),
    )
    _expected_result = {
        'Ng': np.array([
            [-1.6, 0.0, 0.5],
            [0.0, 1.6, 0.5],
            [0.0, -1.6, 0.5],
            [1.6, 0.0, 0.5],
        ], "f4"),
        'geomID': np.array([0, 0, 0, 0], "i4"),
        'primID': np.array([3, 4, 2, 5], "i4"),
        'tfar': 1.2 * np.ones((4, ), "f4"),
        'u': np.array([0.5, 0.5, 0.25, 0.25], "f4"),
        'v': np.array([0.25, 0.25, 0.5, 0.5], "f4"),
    }

    @pytest.fixture(scope="class")
    def intersection_width(self, intersection_radius):
        return np.sqrt(2 * (intersection_radius * intersection_radius))

    @pytest.fixture(scope="class")
    def intersection_radius(self):
        return 0.25

    @pytest.fixture(scope="class")
    def camera_distance(self):
        return 4.0

    @pytest.fixture(scope="class")
    def fov_width(self, intersection_width, camera_distance):
        return np.degrees(
            2.0 * sun_calc.stable_arctan((intersection_width / 2.0)
                                         / camera_distance))

    @pytest.fixture(scope="class")
    def fov_radius(self, intersection_radius, camera_distance):
        return np.degrees(
            sun_calc.stable_arctan(intersection_radius
                                   / camera_distance))

    @pytest.fixture(scope="class")
    def angle_side(self, scene_pyramid):
        r"""Elevation angle of pyramid side from horizontal xy plane."""
        return np.degrees(sun_calc.stable_arctan(1.6 / 0.5))

    @pytest.fixture(scope="class")
    def reorder_rays(self):
        return self._reorder_rays

    @pytest.fixture(scope="class")
    def instance_kws(self):
        return self._instance_kws

    @pytest.fixture(scope="class")
    def instance(self, instance_kws, reorder_rays):
        out = self.cls(**instance_kws)
        # print(out.origins)
        # print(out.directions)
        # if reorder_rays is not None:
        #     print("RE-ORDER")
        #     print(out.directions[np.argsort(reorder_rays)])
        return out

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
    def expected_bounces(self, expected_bounces_base):
        return expected_bounces_base

    @pytest.fixture(scope="class")
    def expected_bounce_factors(self, instance):
        a = 5.6939501e-01
        b = 8.2206404e-01
        c = 0.0
        d = b
        e = a
        return (a, b, c, d, e)

    @pytest.fixture(scope="class")
    def expected_bounces_base(self, get_bounces_empty, expected_result,
                              instance, reorder_rays,
                              expected_bounce_factors):
        out = get_bounces_empty(6)
        if reorder_rays is None:
            reorder_idx = np.arange(out['nbounce'].shape[0], dtype="i4")
        else:
            reorder_idx = np.argsort(reorder_rays)
        out['nbounce'][:] = 6
        out['Ng'][:, 1, 2] = -1
        for i in range(expected_result['Ng'].shape[0]):
            out['Ng'][i, 2, :] = expected_result['Ng'][i]
            out['Ng'][i, 4, 2] = expected_result['Ng'][i, 2]
            out['Ng'][i, 4, :2] = -expected_result['Ng'][i, :2]
        out['geomID'][:, (1, 2, 4)] = 0
        out['primID'][(0, 2), 1] = 0
        out['primID'][(1, 3), 1] = 1
        out['primID'][:, 2] = expected_result['primID']
        out['primID'][:, 4] = expected_result['primID'][::-1]
        out['tfar'][:, (1, 2)] = 7.9989994e-01
        out['tfar'][:, 4] = 6.0501444e-01
        out['u'] = np.array([
            [0.0, 0.25, 0.5, 0.0, 0.09454914182424545, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.09454912, 0.0],
            [0.0, 0.25, 0.25, 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.25, 0.0, 0.81090176, 0.0],
        ], "f4")
        out['v'] = np.array([
            [0.0, 0.5, 0.25, 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.25, 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.5, 0.0, 0.09454914182424545, 0.0],
            [0.0, 0.25, 0.5, 0.0, 0.09454911947250366, 0.0],
        ], "f4")
        out['power'][:] = np.array([
            0.5, 0.25, 0.125, 0.0625, 0.0625, 0.03125,
        ], dtype="f4")

        a, b, c, d, e = expected_bounce_factors[:]
        out['ray_dir'][:, 1, :] = instance.directions[reorder_idx, ...]
        out['ray_dir'][:, 3, :] = instance.directions[reorder_idx, ...]
        out['ray_dir'][:, 2, 2] = -instance.directions[reorder_idx, 2]
        out['ray_dir'][:, 5, 2] = -instance.directions[reorder_idx, 2]
        out['ray_dir'][:, 0, 2] = -b
        out['ray_dir'][:, 4, 2] = d

        out['ray_dir'][0, (2, 5), 0] = -c
        out['ray_dir'][1, (2, 5), 1] = c
        out['ray_dir'][2, (2, 5), 1] = -c
        out['ray_dir'][3, (2, 5), 0] = c

        out['ray_dir'][0, 0, 0] = -a
        out['ray_dir'][0, 4, 0] = e

        out['ray_dir'][1, 0, 1] = a
        out['ray_dir'][1, 4, 1] = -e

        out['ray_dir'][2, 0, 1] = -a
        out['ray_dir'][2, 4, 1] = e

        out['ray_dir'][3, 0, 0] = a
        out['ray_dir'][3, 4, 0] = -e

        return out

    @pytest.fixture(scope="class")
    def expected_result_sorted(self, expected_result, reorder_rays):
        if reorder_rays is None:
            return expected_result
        out = copy.deepcopy(expected_result)
        for k in list(out.keys()):
            out[k] = out[k][reorder_rays, ...]
        return out

    @pytest.fixture(scope="class")
    def expected_bounces_sorted(self, expected_bounces, reorder_rays):
        if reorder_rays is None:
            return expected_bounces
        out = copy.deepcopy(expected_bounces)
        for k in list(out.keys()):
            out[k] = out[k][reorder_rays, ...]
        return out

    def test_compute_distance(self, instance, scene_pyramid,
                              expected_result_sorted, assert_allclose):
        r"""Test calculation of travel distance to scene."""
        actual = instance.compute_distance(scene_pyramid)
        assert_allclose(actual, expected_result_sorted['tfar'])

    def test_compute_count(self, instance, scene_pyramid,
                           expected_result_sorted,
                           assert_dicts_allclose):
        r"""Test calculation of intersections with scene."""
        actual = instance.compute_count(scene_pyramid)
        assert_dicts_allclose(actual, expected_result_sorted)

    def test_compute_count_multibounce(self, instance,
                                       scene_pyramid,
                                       expected_result_sorted,
                                       expected_bounces_sorted,
                                       assert_dicts_allclose):
        r"""Test calculation of intersections with scene."""
        actual = instance.compute_count(
            scene_pyramid, multibounce=True, power_threshold=0.1)
        assert_dicts_allclose(actual, expected_result_sorted,
                              ignore_keys=['bounces'])
        assert 'bounces' in actual
        assert_dicts_allclose(actual['bounces'], expected_bounces_sorted,
                              rtol=1e-6)


class TestOrthographicRayBlaster(TestRayBlaster):
    r"""Tests for RayBlaster class."""

    cls = blaster.OrthographicRayBlaster
    _instance_kws = dict(
        nx=2, ny=2,
    )

    @pytest.fixture(scope="class")
    def instance_kws(self, scene_pyramid, intersection_width):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            width=intersection_width,
            height=intersection_width,
            forward=-scene_pyramid.up,
            center=(scene_pyramid.ground + 2.0 * scene_pyramid.up),
            up=scene_pyramid.north,
        )
        return out


class TestProjectionRayBlaster(TestRayBlaster):
    r"""Tests for ProjectionRayBlaster class."""

    cls = blaster.ProjectionRayBlaster
    _instance_kws = dict(
        nx=2, ny=2,
    )
    _expected_result = dict(
        TestOrthographicRayBlaster._expected_result,
        tfar=1.0019510984420776 * np.ones((4, ), "f4"),
        u=np.array([0.625, 0.625, 0.1875, 0.1875], "f4"),
        v=np.array([0.1875, 0.1875, 0.625, 0.625], "f4"),
    )

    @pytest.fixture(scope="class")
    def instance_kws(self, scene_pyramid, intersection_width,
                     fov_width):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            fov_width=fov_width,
            fov_height=fov_width,
            width=intersection_width / 2,
            height=intersection_width / 2,
            forward=-scene_pyramid.up,
            center=(scene_pyramid.ground + 2.0 * scene_pyramid.up),
            up=scene_pyramid.north,
        )
        return out

    @pytest.fixture(scope="class")
    def expected_bounce_factors(self, instance):
        a = 5.1700723e-01
        b = 8.5598105e-01
        c = 6.2378287e-02
        d = 7.8494531e-01
        e = 6.1956513e-01
        return (a, b, c, d, e)

    @pytest.fixture(scope="class")
    def expected_bounces(self, expected_bounces_base):
        out = copy.deepcopy(expected_bounces_base)

        out['tfar'][:, 1] = 1.0018513
        out['tfar'][:, 2] = 0.6678675
        out['tfar'][:, 4] = 0.6743825

        out['u'][:2, 2] = 0.4166667
        out['u'][2:, 2] = 0.2916667
        out['u'][:2, 4] = 0.1262192
        out['u'][2:, 4] = 0.7475615

        out['v'][:2, 2] = 0.2916667
        out['v'][2:, 2] = 0.4166666
        out['v'][:2, 4] = 0.7475615
        out['v'][2:, 4] = 0.1262192

        return out


class TestSphericalRayBlaster(TestProjectionRayBlaster):

    cls = blaster.SphericalRayBlaster
    _instance_kws = dict(
        dont_include_center=True,
        fov_width=np.float32(270),
        nx=4, ny=1,
    )
    _reorder_rays = np.array([3, 2, 0, 1], "i4")

    @pytest.fixture(scope="class")
    def reorder_rays(self, instance_kws):
        if instance_kws.get('dont_include_center', False):
            return self._reorder_rays
        return np.hstack([np.array([0], "i4"), self._reorder_rays + 1])

    @pytest.fixture(scope="class")
    def instance_kws(self, scene_pyramid, fov_radius):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            fov_height=fov_radius,
            forward=-scene_pyramid.up,
            center=(scene_pyramid.ground + 4.0 * scene_pyramid.up),
            up=scene_pyramid.north,
        )
        return out

    @pytest.fixture(scope="class")
    def expected_result(self, instance_kws, intersection_radius):
        out = copy.deepcopy(self._expected_result)
        out['tfar'][:] += np.float32(
            np.sqrt(4 + (intersection_radius / 2) ** 2))
        if not instance_kws.get('dont_include_center', False):
            out['tfar'] = np.hstack([np.array([2.4], "f4"), out['tfar']])
            out['Ng'] = np.vstack([
                np.array([0.0, 1.6, 0.5], "f4"),
                out['Ng']
            ])
            out['geomID'] = np.hstack([0, out['geomID']])
            out['primID'] = np.hstack([4, out['primID']])
            out['u'] = np.hstack([1.0, out['u']])
            out['v'] = np.hstack([0.0, out['v']])
        return out


class TestSunRayBlaster(TestOrthographicRayBlaster):
    r"""Tests for SunRayBlaster."""

    cls = blaster.SunRayBlaster

    @pytest.fixture(scope="class")
    def instance_kws(self, location_champaign, altitude_champaign,
                     datetime_champaign, scene_pyramid,
                     intersection_width):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            width=intersection_width,
            height=intersection_width,
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
    def expected_result(self):
        out = copy.deepcopy(self._expected_result)
        out['tfar'] = np.array([
            1.3942306, 1.5534554, 1.394803, 1.5385036
        ], "f4")
        out['u'] = np.array([
            0.4938756, 0.46215066, 0.26069573, 0.29107162
        ], "f4")
        out['v'] = np.array([
            0.23884138, 0.27587134, 0.493533, 0.4711003
        ], "f4")
        return out

    @pytest.fixture(scope="class")
    def expected_bounces(self, expected_bounces_base):
        out = copy.deepcopy(expected_bounces_base)
        out['Ng'][0, 2, :2] = [0.0, 1.6]
        out['Ng'][0, 4, :2] = [0.0, -1.6]
        out['Ng'][1, 4, :2] = [1.6, 0.0]
        out['Ng'][2, 2, :2] = [1.6, 0.0]
        out['Ng'][3, 4, :2] = [0.0, 1.6]
        out['primID'][:, 1] = 1
        out['primID'][:2, 2] = 4
        out['primID'][2:, 2] = 5
        out['primID'][:2, 4] = [2, 5]
        out['primID'][2:, 4] = 4
        out['ray_dir'][:, (2, 5), :] = [
            0.19461885, 0.21193677, 0.9577089]

        out['ray_dir'][0, 0, :] = [-0.7053038, 0.21193677, -0.67648304]
        out['ray_dir'][0, 4, :] = [0.19461885, -0.71954024, 0.66662234]

        out['ray_dir'][1, 0, :] = [0.19461885, 0.3710891, -0.9079738]
        out['ray_dir'][1, 4, :] = [0.19461885, -0.71954024, 0.66662234]

        out['ray_dir'][2, 0, :] = [0.19461885, -0.71954024, -0.66662234]
        out['ray_dir'][2, 4, :] = [-0.7053038, 0.21193677, 0.67648304]

        out['ray_dir'][3, 0, :] = [0.38532552, 0.21193677, -0.8981131]
        out['ray_dir'][3, 4, :] = [-0.7053038, 0.21193677, 0.67648304]

        out['tfar'][0, (1, 2, 4)] = [
            8.2499510e-01, 6.6370875e-01, 6.4942205e-01]
        out['tfar'][1, (1, 2, 4)] = [
            7.7199394e-01, 1.3181996e-01, 6.8892235e-01]
        out['tfar'][2, (1, 2, 4)] = [
            8.2442266e-01, 6.7223883e-01, 5.5447680e-01]
        out['tfar'][3, (1, 2, 4)] = [
            7.8694570e-01, 1.6668625e-01, 5.2267307e-01]

        out['u'][0, (1, 2, 4)] = [0.592483, 0.397335, 0.3291404]
        out['u'][1, (1, 2, 4)] = [0.342789, 0.07896312, 0.2817207]
        out['u'][2, (1, 2, 4)] = [0.33207, 0.362786, 0.636917]
        out['u'][3, (1, 2, 4)] = [0.08237622, 0.6788575, 0.32086235]

        out['v'][0, (1, 2, 4)] = [0.06816402, 0.3380392, 0.66795087]
        out['v'][1, (1, 2, 4)] = [0.58977044, 0.64340335, 0.36603665]
        out['v'][2, (1, 2, 4)] = [0.08944302, 0.40244052, 0.08917636]
        out['v'][3, (1, 2, 4)] = [0.6110496, 0.09983292, 0.42093852]

        return out
