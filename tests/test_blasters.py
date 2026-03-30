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
        return np.degrees(2.0 * np.arctan((intersection_width / 2.0)
                                          / camera_distance))

    @pytest.fixture(scope="class")
    def fov_radius(self, intersection_radius, camera_distance):
        return np.degrees(np.arctan(intersection_radius
                                    / camera_distance))

    @pytest.fixture(scope="class")
    def angle_side(self, scene_pyramid):
        r"""Elevation angle of pyramid side from horizontal xy plane."""
        return np.degrees(np.arctan(1.6 / 0.5))

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
    def expected_bounces(self, expected_bounces_base):
        return expected_bounces_base

    @pytest.fixture(scope="class")
    def expected_bounces_base(self, get_bounces_empty, expected_result,
                              instance, reorder_rays):
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
            [0.0, 0.25, 0.5, 0.0, 0.09454912, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.09454912, 0.0],
            [0.0, 0.25, 0.25, 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.25, 0.0, 0.81090176, 0.0],
        ], "f4")
        out['v'] = np.array([
            [0.0, 0.5, 0.25, 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.25, 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.5, 0.0, 0.09454912, 0.0],
            [0.0, 0.25, 0.5, 0.0, 0.09454914, 0.0],
        ], "f4")
        out['power'][:] = np.array([
            0.5, 0.25, 0.125, 0.0625, 0.0625, 0.03125,
        ], dtype="f4")
        out['ray_dir'][:, 1, :] = instance.directions[reorder_idx, ...]
        out['ray_dir'][:, 3, :] = instance.directions[reorder_idx, ...]
        out['ray_dir'][:, 2, 2] = -instance.directions[reorder_idx, 2]
        out['ray_dir'][:, 5, 2] = -instance.directions[reorder_idx, 2]
        out['ray_dir'][:, 0, 2] = -8.2206404e-01
        out['ray_dir'][:, 4, 2] = 8.2206404e-01

        out['ray_dir'][0, 0, 0] = -5.6939501e-01
        out['ray_dir'][0, 0, 1] = -1.1689008e-16
        out['ray_dir'][0, 4, 0] = 5.6939501e-01
        out['ray_dir'][0, 4, 1] = 1.1689008e-16

        out['ray_dir'][1, 0, 0] = -1.1689008e-16
        out['ray_dir'][1, 0, 1] = 5.6939501e-01
        out['ray_dir'][1, 4, 0] = 1.1689008e-16
        out['ray_dir'][1, 4, 1] = -5.6939501e-01

        out['ray_dir'][2, 0, 0] = 1.1689008e-16
        out['ray_dir'][2, 0, 1] = -5.6939501e-01
        out['ray_dir'][2, 4, 0] = -1.1689008e-16
        out['ray_dir'][2, 4, 1] = 5.6939501e-01

        out['ray_dir'][3, 0, 0] = 5.6939501e-01
        out['ray_dir'][3, 0, 1] = 1.1689008e-16
        out['ray_dir'][3, 4, 0] = -5.6939501e-01
        out['ray_dir'][3, 4, 1] = -1.1689008e-16

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
                              expected_result_sorted):
        r"""Test calculation of travel distance to scene."""
        actual = instance.compute_distance(scene_pyramid)
        assert_almost_equal(actual, expected_result_sorted['tfar'],
                            decimal=6)

    def test_compute_count(self, instance, scene_pyramid,
                           expected_result_sorted,
                           assert_dicts_almost_equal):
        r"""Test calculation of intersections with scene."""
        actual = instance.compute_count(scene_pyramid)
        assert_dicts_almost_equal(actual, expected_result_sorted,
                                  decimal=6)

    def test_compute_count_multibounce(self, instance_multibounce,
                                       scene_pyramid,
                                       expected_result_sorted,
                                       expected_bounces_sorted,
                                       assert_dicts_almost_equal):
        r"""Test calculation of intersections with scene."""
        actual = instance_multibounce.compute_count(scene_pyramid)
        assert_dicts_almost_equal(actual, expected_result_sorted,
                                  ignore_keys=['bounces'], decimal=6)
        assert 'bounces' in actual
        assert_dicts_almost_equal(actual['bounces'],
                                  expected_bounces_sorted, decimal=6)


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
    def expected_bounces(self, expected_bounces_base):
        out = copy.deepcopy(expected_bounces_base)
        out['ray_dir'][:, 0, 2] = -8.5598105e-01
        out['ray_dir'][:, 4, 2] = 7.8494531e-01
        a = 5.1700723e-01
        b = 1.1894101e-16
        c = 6.2378287e-02
        d = 7.6391370e-18
        e = 6.1956513e-01
        f = 1.2202302e-16

        out['ray_dir'][0, 0, :2] = [-a, -b]
        out['ray_dir'][0, 2, :2] = [-c, d]
        out['ray_dir'][0, 4, :2] = [e,  f]
        out['ray_dir'][0, 5, :2] = [-c, d]

        out['ray_dir'][1, 0, :2] = [-b, a]
        out['ray_dir'][1, 2, :2] = [d, c]
        out['ray_dir'][1, 4, :2] = [f, -e]
        out['ray_dir'][1, 5, :2] = [d, c]

        out['ray_dir'][2, 0, :2] = [b, -a]
        out['ray_dir'][2, 2, :2] = [-d, -c]
        out['ray_dir'][2, 4, :2] = [-f, e]
        out['ray_dir'][2, 5, :2] = [-d, -c]

        out['ray_dir'][3, 0, :2] = [a, b]
        out['ray_dir'][3, 2, :2] = [c, -d]
        out['ray_dir'][3, 4, :2] = [-e, -f]
        out['ray_dir'][3, 5, :2] = [c, -d]

        out['tfar'][:, 1] = 1.0018513e+00
        out['tfar'][:, 2] = 6.6786748e-01
        out['tfar'][(0, 2), 4] = 6.7438251e-01
        out['tfar'][(1, 3), 4] = 6.7438257e-01

        out['u'][(0, 2, 3), 1] = 0.25
        out['u'][1, 1] = 0.5
        out['u'][:2, 2] = 0.41666666
        out['u'][2:, 2] = 0.29166672
        out['u'][:2, 4] = 0.1262192
        out['u'][2:, 4] = 0.7475615
        out['v'][0, 1] = 0.5
        out['v'][1:, 1] = 0.25
        out['v'][:2, 2] = 0.29166672
        out['v'][2:, 2] = 0.41666666
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
    def expected_result(self, instance_kws, fov_radius):
        out = copy.deepcopy(self._expected_result)
        out['tfar'][:] += 2 / np.cos(np.radians(fov_radius))
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
            1.3942287, 1.5534511, 1.3948011, 1.5385
        ], "f4")
        out['u'] = np.array([
            0.4938758, 0.4621518, 0.2606957, 0.2910709
        ], "f4")
        out['v'] = np.array([
            0.2388414, 0.2758704, 0.4935333, 0.4711011
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
            0.19461735, 0.21193513, 0.95770955]

        out['ray_dir'][0, 0, :] = [-0.70530295, 0.21193513, -0.67648447]
        out['ray_dir'][0, 4, :] = [0.19461735, -0.7195393, 0.6666238]

        out['ray_dir'][1, 0, :] = [0.19461735, 0.3710908, -0.9079734]
        out['ray_dir'][1, 4, :] = [0.19461735, -0.7195393, 0.6666238]

        out['ray_dir'][2, 0, :] = [0.19461735, -0.7195392, -0.66662383]
        out['ray_dir'][2, 4, :] = [-0.70530295, 0.21193513, 0.67648447]

        out['ray_dir'][3, 0, :] = [0.38532713, 0.21193513, -0.8981127]
        out['ray_dir'][3, 4, :] = [-0.70530295, 0.21193513, 0.67648447]

        out['tfar'][0, (1, 2, 4)] = [
            8.2499504e-01, 6.6371316e-01, 6.4941913e-01]
        out['tfar'][1, (1, 2, 4)] = [
            7.7199513e-01, 1.3182324e-01, 6.8892407e-01]
        out['tfar'][2, (1, 2, 4)] = [
            8.2442254e-01, 6.7224294e-01, 5.5447859e-01]
        out['tfar'][3, (1, 2, 4)] = [
            7.8694624e-01, 1.6668956e-01, 5.2267480e-01]

        out['u'][0, (1, 2, 4)] = [0.5924843, 0.3973376, 0.3291364]
        out['u'][1, (1, 2, 4)] = [0.3427906, 0.0789651, 0.2817175]
        out['u'][2, (1, 2, 4)] = [0.3320713, 0.3627833, 0.6369207]
        out['u'][3, (1, 2, 4)] = [0.08237763, 0.67885536, 0.32086563]

        out['v'][0, (1, 2, 4)] = [0.0681615, 0.3380364, 0.6679531]
        out['v'][1, (1, 2, 4)] = [0.5897675, 0.6434013, 0.36604]
        out['v'][2, (1, 2, 4)] = [0.0894406, 0.4024433, 0.0891721]
        out['v'][3, (1, 2, 4)] = [0.61104655, 0.09983501, 0.4209351]

        return out
