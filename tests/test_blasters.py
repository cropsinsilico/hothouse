import pytest
import copy
import numpy as np
import traitlets
from hothouse import blaster, sun_calc


def test_sun_blaster(location_champaign, altitude_champaign,
                     datetime_champaign, geometry_scene,
                     assert_allclose, tolerances_solar):
    r"""Test creation & use of solar blaster."""
    nx = 512
    ny = 512
    date = datetime_champaign("sunrise")
    instance = geometry_scene("soy")
    rb = instance.get_sun_blaster(*location_champaign, date,
                                  altitude=altitude_champaign,
                                  nx=nx, ny=ny)
    assert_allclose(rb.solar_altitude, 7.807668468792781,
                    **tolerances_solar)
    assert_allclose(rb.solar_distance, 694.869384765625,
                    **tolerances_solar)
    assert_allclose(
        rb.center,
        np.array([574.7942504882812, 254.042, 532.0391845703125], "f8"),
        **tolerances_solar
    )


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
        ], dtype="f8"),
        directions=np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ], dtype="f8"),
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
    _expected_attributes = {
        'reflectance': 0.5,
        'transmittance': 0.25,
    }
    _flux_density_result = 0.08896797

    @pytest.fixture(scope="class")
    def instance_type(self):
        return "base"

    @pytest.fixture(scope="class")
    def skip_if_not_base(self, instance_type):
        if instance_type != 'base':
            pytest.skip("Only enabled for base instance")

    @pytest.fixture(scope="class", params=["pyramid"])
    def scene_geometry(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def scene_instance(self, scene_geometry, geometry_scene):
        return geometry_scene(scene_geometry)

    @pytest.fixture
    def flux_density_result(self):
        r"""float: Value for compute_flux_density test."""
        return self._flux_density_result

    @pytest.fixture(scope="class")
    def tolerances(self):
        return {}

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
    def angle_side(self, scene_instance):
        r"""Elevation angle of pyramid side from horizontal xy plane."""
        return np.degrees(sun_calc.stable_arctan(1.6 / 0.5))

    @pytest.fixture(scope="class")
    def reorder_rays(self):
        return self._reorder_rays

    @pytest.fixture(scope="class")
    def instance_kws_base(self):
        return self._instance_kws

    @pytest.fixture(scope="class")
    def instance_kws(self, instance_type, instance_kws_base):
        if instance_type == "base":
            return instance_kws_base
        raise NotImplementedError(instance_type)

    @pytest.fixture(scope="class")
    def instance_base(self, instance_kws_base, reorder_rays):
        out = self.cls(**instance_kws_base)
        # print(out.origins)
        # print(out.directions)
        # if reorder_rays is not None:
        #     print("RE-ORDER")
        #     print(out.directions[np.argsort(reorder_rays)])
        return out

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
    def expected_attributes(self, instance):
        out = {
            k: np.empty(instance.nray)
            for k in self._expected_attributes.keys()
        }
        for k, v in self._expected_attributes.items():
            out[k].fill(v)
        return out

    @pytest.fixture(scope="class")
    def get_results_empty(self, instance):

        nray = instance.nray

        def _get_results_empty(nbounce=None):
            flatten = (nbounce is None)
            if nbounce is None:
                nbounce = 1
            out = {
                'Ng': np.empty((nray, nbounce, 3), "f4"),
                'geomID': np.empty((nray, nbounce), "i4"),
                'primID': np.empty((nray, nbounce), "i4"),
                'tfar': np.empty((nray, nbounce), "f4"),
                'u': np.empty((nray, nbounce), "f4"),
                'v': np.empty((nray, nbounce), "f4"),
            }
            for k, v in out.items():
                v.fill(self.cls._null_field_values.get(k, 0))
            if flatten:
                for k in list(out.keys()):
                    out[k] = out[k].reshape(nray, -1)
            else:
                out.update(
                    nbounce=np.zeros((nray, ), "i4"),
                    ray_intensity=np.zeros((nray, nbounce), "f8"),
                    ray_dir=np.zeros((nray, nbounce, 3), "f8"),
                    transmittance=np.zeros((nray, nbounce), "f8"),
                    reflectance=np.zeros((nray, nbounce), "f8"),
                )
            return out

        return _get_results_empty

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
    def expected_bounces_base(self, get_results_empty, expected_result,
                              instance, reorder_rays,
                              expected_bounce_factors):
        nbounce = 3
        out = get_results_empty(nbounce)
        if reorder_rays is None:
            reorder_idx = np.arange(instance.nray, dtype="i4")
        else:
            reorder_idx = np.argsort(reorder_rays)
        out['nbounce'][:] = nbounce
        out['Ng'][:, 1, 2] = -1
        for i in range(expected_result['Ng'].shape[0]):
            out['Ng'][i, 2, :] = expected_result['Ng'][i]
            # out['Ng'][i, 4, 2] = expected_result['Ng'][i, 2]
            # out['Ng'][i, 4, :2] = -expected_result['Ng'][i, :2]
        out['geomID'][:, (1, 2)] = 0
        # out['geomID'][:, (1, 2, 4)] = 0
        out['primID'][(0, 2), 1] = 0
        out['primID'][(1, 3), 1] = 1
        out['primID'][:, 2] = expected_result['primID']
        # out['primID'][:, 4] = expected_result['primID'][::-1]
        out['tfar'][:, (1, 2)] = 7.9989994e-01
        # out['tfar'][:, 4] = 6.0501444e-01
        out['u'][:4, ...] = np.array([
            [0.0, 0.25, 0.5],  # , 0.0, 0.09454914182424545, 0.0],
            [0.0, 0.5, 0.5],  # , 0.0, 0.09454912, 0.0],
            [0.0, 0.25, 0.25],  # , 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.25],  # , 0.0, 0.81090176, 0.0],
        ], "f4")
        out['v'][:4, ...] = np.array([
            [0.0, 0.5, 0.25],  # , 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.25],  # , 0.0, 0.81090176, 0.0],
            [0.0, 0.25, 0.5],  # , 0.0, 0.09454914182424545, 0.0],
            [0.0, 0.25, 0.5],  # , 0.0, 0.09454911947250366, 0.0],
        ], "f4")
        out['ray_intensity'][:] = np.array([
            0.125, 0.0625, 0.03125,  # 0.0625, 0.0625, 0.03125,
        ], dtype="f8")

        a, b, c, d, e = expected_bounce_factors[:]
        out['ray_dir'][:, 1, :] = instance.directions[reorder_idx, ...]
        # out['ray_dir'][:, 3, :] = instance.directions[reorder_idx, ...]
        out['ray_dir'][:, 2, 2] = -instance.directions[reorder_idx, 2]
        # out['ray_dir'][:, 5, 2] = -instance.directions[reorder_idx, 2]
        out['ray_dir'][:, 0, 2] = -b
        # out['ray_dir'][:, 4, 2] = d

        out['ray_dir'][0, 2, 0] = -c
        out['ray_dir'][1, 2, 1] = c
        out['ray_dir'][2, 2, 1] = -c
        out['ray_dir'][3, 2, 0] = c
        # out['ray_dir'][0, (2, 5), 0] = -c
        # out['ray_dir'][1, (2, 5), 1] = c
        # out['ray_dir'][2, (2, 5), 1] = -c
        # out['ray_dir'][3, (2, 5), 0] = c

        out['ray_dir'][0, 0, 0] = -a
        # out['ray_dir'][0, 4, 0] = e

        out['ray_dir'][1, 0, 1] = a
        # out['ray_dir'][1, 4, 1] = -e

        out['ray_dir'][2, 0, 1] = -a
        # out['ray_dir'][2, 4, 1] = e

        out['ray_dir'][3, 0, 0] = a
        # out['ray_dir'][3, 4, 0] = -e

        idx_hits = (out['primID'] != -1)
        out['reflectance'][idx_hits] = 0.5
        out['transmittance'][idx_hits] = 0.25

        return out

    @pytest.fixture(scope="class")
    def reorder_result(self, reorder_rays):

        def _reorder_result(result):
            if reorder_rays is None:
                return result
            out = copy.deepcopy(result)
            for k in list(out.keys()):
                out[k] = out[k][reorder_rays, ...]
            return out

        return _reorder_result

    @pytest.fixture(scope="class")
    def expected_result_sorted(self, reorder_result, expected_result):
        return reorder_result(expected_result)

    @pytest.fixture(scope="class")
    def expected_attributes_sorted(self, reorder_result,
                                   expected_attributes):
        return reorder_result(expected_attributes)

    @pytest.fixture(scope="class")
    def expected_bounces_sorted(self, reorder_result, expected_bounces):
        return reorder_result(expected_bounces)

    def test_attributes(self, instance, assert_allclose):
        r"""Test attributes."""
        assert_allclose(instance.intensity, 1.0)

    def test_missing_intensity(self, instance_kws_base, assert_allclose,
                               skip_if_not_base):
        r"""Check alternate method of calculating default values."""
        if 'origins' in instance_kws_base:
            N = instance_kws_base['origins'].shape[0]
        else:
            N = instance_kws_base['nx'] * instance_kws_base['ny']
        x = self.cls(
            ray_intensity=np.ones((N, ), "f8"),
            **instance_kws_base
        )
        assert_allclose(x.intensity, N)
        assert_allclose(x.nray, N)

    def test_compute_distance(self, instance, scene_instance,
                              expected_result_sorted,
                              assert_allclose, tolerances):
        r"""Test calculation of travel distance to scene."""
        actual = instance.compute_distance(scene_instance)
        assert_allclose(actual, expected_result_sorted['tfar'],
                        **tolerances)

    def test_compute_occluded(self, instance, scene_instance,
                              expected_result_sorted,
                              assert_allclose):
        r"""Test calculation of occluded rays."""
        actual = instance.compute_occluded(scene_instance)
        expected = -1 * np.ones(instance.nray, "i4")
        expected[expected_result_sorted['primID'] != -1] = 0
        assert_allclose(actual, expected)

    def test_compute_intersect(self, instance, scene_instance,
                               expected_result_sorted,
                               assert_allclose):
        r"""Test calculation of rays intersection."""
        actual = instance.compute_intersect(scene_instance)
        assert_allclose(actual, expected_result_sorted['primID'])

    @pytest.mark.parametrize("include_attributes", [
        False,
        True,
        ['dummy'],
    ])
    def test_compute_count(self, include_attributes,
                           instance, scene_instance,
                           expected_result_sorted,
                           assert_nested_allclose, tolerances):
        r"""Test calculation of intersections with scene."""
        actual = instance.compute_count(
            scene_instance, include_attributes=include_attributes,
        )
        if include_attributes is True:
            extra_attributes = ['reflectance', 'transmittance']
        elif isinstance(include_attributes, list):
            extra_attributes = include_attributes
        else:
            extra_attributes = []
        assert_nested_allclose(
            actual, expected_result_sorted,
            ignore_keys=extra_attributes,
            **tolerances
        )
        if include_attributes:
            return
        bounce = instance.bounce(
            actual, ray_intensity_threshold_abs=(
                0.1 * instance.ray_intensity[0]
            ),
        )
        assert_nested_allclose(bounce.ray_intensity_threshold_rel, 0.1)
        bounce.ray_intensity_threshold_rel = 0.5
        assert_nested_allclose(bounce.ray_intensity_threshold_abs,
                               0.5 * instance.ray_intensity[0])
        bounce = instance.bounce(actual)
        assert bounce.ray_intensity_threshold_rel == 0.001

    @pytest.mark.parametrize("include_attributes", [
        False,
        True,
        ['dummy'],
    ])
    def test_compute_count_multibounce(self, include_attributes,
                                       instance, scene_instance,
                                       expected_result_sorted,
                                       expected_bounces_sorted,
                                       expected_attributes_sorted,
                                       assert_nested_allclose,
                                       tolerances, tolerances_bounces):
        r"""Test calculation of intersections with scene."""
        actual = instance.compute_count(
            scene_instance, multibounce=True,
            ray_intensity_threshold_rel=0.1,
            include_attributes=include_attributes,
        )
        extra_attributes = []
        expected_attributes = expected_attributes_sorted.copy()
        if isinstance(include_attributes, list):
            extra_attributes = include_attributes
            for k in include_attributes:
                if k in expected_attributes:
                    continue
                expected_attributes[k] = np.zeros(instance.nray, "f8")
        attributes = (
            list(expected_attributes_sorted.keys())
            + extra_attributes
        )
        assert_nested_allclose(
            actual, expected_result_sorted,
            ignore_keys=['bounces'] + attributes,
            **tolerances
        )
        assert_nested_allclose(
            actual, expected_attributes, only_keys=attributes,
            **tolerances
        )
        assert 'bounces' in actual
        assert_nested_allclose(
            actual['bounces'], expected_bounces_sorted,
            ignore_keys=extra_attributes,
            **tolerances_bounces
        )
        for k in extra_attributes:
            assert_nested_allclose(
                actual['bounces'][k],
                np.zeros(actual['bounces']['transmittance'].shape, "f8"),
            )

    def test_compute_count_attributes(self, instance, scene_instance,
                                      expected_result_sorted,
                                      expected_attributes_sorted,
                                      assert_nested_allclose, tolerances):
        actual = instance.compute_count(scene_instance,
                                        include_attributes=True)
        assert_nested_allclose(
            actual, expected_result_sorted,
            ignore_keys=list(expected_attributes_sorted.keys()),
            **tolerances
        )
        assert_nested_allclose(
            actual, expected_attributes_sorted,
            only_keys=list(expected_attributes_sorted.keys()),
            **tolerances
        )

    def test_compute_flux_density(self, instance, scene_instance,
                                  assert_nested_allclose,
                                  expected_result_sorted,
                                  tolerances, flux_density_result):
        r"""Test compute_flux_density."""
        actual = instance.compute_flux_density(scene_instance, instance)
        if isinstance(flux_density_result, np.ndarray):
            expected = flux_density_result
        else:
            expected = np.empty((instance.nray, ), dtype="f8")
            expected.fill(flux_density_result)
        assert_nested_allclose(actual, expected, **tolerances)


@pytest.mark.parametrize("instance_type", ["base", "periodic"],
                         scope="class")
class TestOrthographicRayBlaster(TestRayBlaster):
    r"""Tests for OrthographicRayBlaster class."""

    cls = blaster.OrthographicRayBlaster
    _instance_kws = dict(
        nx=2, ny=2,
    )
    instance_type = None

    @pytest.fixture(scope="class")
    def instance_kws_base(self, scene_instance, intersection_width):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            width=intersection_width,
            height=intersection_width,
            forward=-scene_instance.up,
            center=(scene_instance.ground + 2.0 * scene_instance.up),
            up=scene_instance.north,
        )
        return out

    def test_attributes(self, instance, instance_kws, assert_allclose):
        r"""Test attributes."""
        if not isinstance(instance, blaster.SunRayBlaster):
            assert_allclose(instance.intensity, 1.0)
            kws2 = copy.deepcopy(instance_kws)
            kws2.pop('forward')
            kws2['east'] = instance.east
            instance2 = self.cls(**kws2)
            assert_allclose(instance2.forward, instance.forward)
            assert_allclose(instance.east, instance2.east)
            kws3 = copy.deepcopy(instance_kws)
            kws3.pop('up')
            kws3['east'] = instance.east
            instance3 = self.cls(**kws3)
            assert_allclose(instance3.up, instance.up)
            assert_allclose(instance.east, instance3.east)
        assert_allclose(instance.intensity,
                        instance.intensity_density
                        * instance.width * instance.height)

    def test_traits_errors(self, skip_if_not_base):
        r"""Test errors raised for invalid traits."""
        with pytest.raises(traitlets.TraitError):
            self.cls().forward

    @pytest.fixture(scope="class")
    def periodic_shift(self, instance_base):
        return (
            - (instance_base.width * (1 + 1 / instance_base.nx)
               * instance_base.east)
            - (instance_base.height * (1 + 1 / instance_base.ny)
               * instance_base.up)
        )

    @pytest.fixture(scope="class")
    def instance_kws(self, instance_type, instance_kws_base,
                     periodic_shift):
        if instance_type == "base":
            return instance_kws_base
        elif instance_type == "periodic":
            out = copy.deepcopy(instance_kws_base)
            out['period'] = np.ones((3, ), dtype="i4")
            for k in ['center', 'ground', 'zenith', 'scene_limits']:
                if k not in out:
                    continue
                out[k] = out[k] + periodic_shift
            return out
        raise NotImplementedError(instance_type)


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
    _flux_density_result = 0.07103577

    @pytest.fixture(scope="class")
    def instance_kws_base(self, scene_instance, intersection_width,
                          fov_width):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            fov_width=fov_width,
            fov_height=fov_width,
            width=intersection_width / 2,
            height=intersection_width / 2,
            forward=-scene_instance.up,
            center=(scene_instance.ground + 2.0 * scene_instance.up),
            up=scene_instance.north,
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
        # out['tfar'][:, 4] = 0.6743825

        out['u'][:2, 2] = 0.4166667
        out['u'][2:, 2] = 0.2916667
        # out['u'][:2, 4] = 0.1262192
        # out['u'][2:, 4] = 0.7475615

        out['v'][:2, 2] = 0.2916667
        out['v'][2:, 2] = 0.4166666
        # out['v'][:2, 4] = 0.7475615
        # out['v'][2:, 4] = 0.1262192

        return out


@pytest.mark.parametrize("instance_type", ["base", "with_center"],
                         scope="class")
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
        return np.hstack([np.array([self._reorder_rays.shape[0]], "i4"),
                          self._reorder_rays])

    @pytest.fixture(scope="class")
    def instance_kws_base(self, scene_instance, fov_radius):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            fov_height=fov_radius,
            forward=-scene_instance.up,
            center=(scene_instance.ground + 4.0 * scene_instance.up),
            up=scene_instance.north,
        )
        return out

    @pytest.fixture(scope="class")
    def instance_kws(self, instance_type, instance_kws_base):
        if instance_type == "base":
            return instance_kws_base
        elif instance_type == "with_center":
            out = copy.deepcopy(instance_kws_base)
            out['dont_include_center'] = False
            return out
        raise NotImplementedError(instance_type)

    @pytest.fixture
    def flux_density_result(self, instance_kws, instance):
        r"""float: Value for compute_flux_density test."""
        if instance_kws.get('dont_include_center', False):
            return self._flux_density_result
        out = np.empty(instance.nray, "f8")
        out.fill(0.05682861)
        out[0] = 0.12800299  # Extra contribution from center ray
        out[4] = 0.12800299  # Extra contribution from center ray
        return out

    @pytest.fixture(scope="class")
    def expected_result(self, instance_kws, intersection_radius):
        out = copy.deepcopy(self._expected_result)
        out['tfar'][:] += np.float32(
            np.sqrt(4 + (intersection_radius / 2) ** 2))
        if instance_kws.get('dont_include_center', False):
            return out
        out['tfar'] = np.hstack([out['tfar'], np.array([2.4], "f4")])
        out['Ng'] = np.vstack([
            out['Ng'],
            np.array([0.0, 1.6, 0.5], "f4"),
        ])
        out['geomID'] = np.hstack([out['geomID'], 0])
        out['primID'] = np.hstack([out['primID'], 4])
        out['u'] = np.hstack([out['u'], 1.0])
        out['v'] = np.hstack([out['v'], 0.0])
        return out

    @pytest.fixture(scope="class")
    def expected_bounces_sorted(self, reorder_result, expected_bounces,
                                instance_kws):
        out = reorder_result(expected_bounces)
        if instance_kws.get('dont_include_center', False):
            return out
        out['primID'][0, 1] = 0
        out['tfar'][0, (1, 2)] = 1.5999
        out['ray_dir'][0, 0, (1, 2)] = [0.56939501, -0.82206404]
        out['ray_intensity'] *= 4.0 / 5.0
        out['reflectance'][0, 1] = 0.5
        out['transmittance'][0, 1] = 0.25
        out['u'][0, 2] = 1.0
        out['v'][0, (1, 2)] = [0.5, 0.0]
        return out


class TestSunRayBlaster(TestOrthographicRayBlaster):
    r"""Tests for SunRayBlaster."""

    cls = blaster.SunRayBlaster
    _flux_density_result = np.array([
        188.442729, 168.678366, 189.284625, 169.520262
    ])

    @pytest.fixture(scope="class")
    def tolerances(self, tolerances_solar):
        return tolerances_solar

    @pytest.fixture(scope="class")
    def instance_kws_base(self, location_champaign, altitude_champaign,
                          datetime_champaign, scene_instance,
                          intersection_width):
        out = copy.deepcopy(self._instance_kws)
        out.update(
            width=intersection_width,
            height=intersection_width,
            latitude=location_champaign[0],
            longitude=location_champaign[1],
            altitude=altitude_champaign,
            date=datetime_champaign("noon"),
            ground=scene_instance.ground,
            zenith=(scene_instance.ground + 2.0 * scene_instance.up),
            north=scene_instance.north,
            scene_limits=scene_instance.limits,
        )
        return out

    @pytest.fixture(scope="class")
    def expected_result(self):
        out = copy.deepcopy(self._expected_result)
        out['tfar'] = np.array([
            1.3942306, 1.5534554, 1.394803, 1.5385036
        ], "f4")
        out['u'] = np.array([
            0.4938758, 0.4621504, 0.26069576, 0.291072
        ], "f4")
        out['v'] = np.array([
            0.23884134, 0.2758717, 0.4935331, 0.47110012
        ], "f4")
        return out

    @pytest.fixture(scope="class")
    def expected_bounces(self, expected_bounces_base):
        out = copy.deepcopy(expected_bounces_base)
        out['Ng'][0, 2, :2] = [0.0, 1.6]
        # out['Ng'][0, 4, :2] = [0.0, -1.6]
        # out['Ng'][1, 4, :2] = [1.6, 0.0]
        out['Ng'][2, 2, :2] = [1.6, 0.0]
        # out['Ng'][3, 4, :2] = [0.0, 1.6]
        out['primID'][:, 1] = 1
        out['primID'][:2, 2] = 4
        out['primID'][2:, 2] = 5
        # out['primID'][:2, 4] = [2, 5]
        # out['primID'][2:, 4] = 4
        out['ray_dir'][:, 2, :] = [
            0.19461885, 0.21193677, 0.9577089]
        # out['ray_dir'][:, (2, 5), :] = [
        #     0.19461885, 0.21193677, 0.9577089]

        out['ray_dir'][0, 0, :] = [-0.7053038, 0.21193677, -0.67648304]
        # out['ray_dir'][0, 4, :] = [0.19461885, -0.71954024, 0.66662234]

        out['ray_dir'][1, 0, :] = [0.19461885, 0.3710891, -0.9079738]
        # out['ray_dir'][1, 4, :] = [0.19461885, -0.71954024, 0.66662234]

        out['ray_dir'][2, 0, :] = [0.19461885, -0.71954024, -0.66662234]
        # out['ray_dir'][2, 4, :] = [-0.7053038, 0.21193677, 0.67648304]

        out['ray_dir'][3, 0, :] = [0.38532552, 0.21193677, -0.8981131]
        # out['ray_dir'][3, 4, :] = [-0.7053038, 0.21193677, 0.67648304]

        idx_hit = (1, 2)
        # idx_hit = (1, 2, 4)

        out['tfar'][0, idx_hit] = [
            8.2499540e-01, 6.6370875e-01]  # , 6.4942205e-01]
        out['tfar'][1, idx_hit] = [
            7.7199340e-01, 1.3182010e-01]  # , 6.8892252e-01]
        out['tfar'][2, idx_hit] = [
            8.2442284e-01, 6.7223877e-01]  # , 5.5447668e-01]
        out['tfar'][3, idx_hit] = [
            7.8694510e-01, 1.6668637e-01]  # , 5.2267313e-01]

        out['u'][0, idx_hit] = [0.592483, 0.397335]  # , 0.3291404]
        out['u'][1, idx_hit] = [0.342789, 0.07896312]  # , 0.2817207]
        out['u'][2, idx_hit] = [0.33207, 0.362786]  # , 0.636917]
        out['u'][3, idx_hit] = [0.08237622, 0.6788575]  # , 0.32086235]

        out['v'][0, idx_hit] = [0.06816418, 0.33803925]  # , 0.66795087]
        out['v'][1, idx_hit] = [0.5897705, 0.64340323]  # , 0.36603677]
        out['v'][2, idx_hit] = [0.0894432, 0.40244046]  # , 0.08917642]
        out['v'][3, idx_hit] = [0.6110496, 0.09983302]  # , 0.42093846]

        out['ray_intensity'][:, 0] = 21.3446598
        out['ray_intensity'][:, 1] = 10.6723299
        out['ray_intensity'][:, 2] = 5.33616491

        return out

    def test_traits_errors(self, instance_kws, datetime_champaign,
                           skip_if_not_base):
        r"""Test errors raised for invalid traits."""
        with pytest.raises(traitlets.TraitError):
            self.cls().forward
        with pytest.raises(traitlets.TraitError):
            kws = instance_kws.copy()
            kws['date'] = datetime_champaign('midnight')
            self.cls(**kws).solar_altitude
        with pytest.raises(traitlets.TraitError):
            kws = instance_kws.copy()
            kws['solar_altitude'] = -50.0
            self.cls(**kws)
        with pytest.raises(traitlets.TraitError):
            kws = instance_kws.copy()
            kws['ray_intensity'] = 1.0
            self.cls(**kws)

    def test_alternate_constructions(self, instance_type, instance_kws,
                                     periodic_shift,
                                     assert_nested_allclose, tolerances):
        # Valid solar altitude
        kws = instance_kws.copy()
        kws['solar_altitude'] = 50.0
        self.cls(**kws)
        # Center from ground
        kws = instance_kws.copy()
        kws['scene_limits'] = None
        inst = self.cls(**kws)
        assert inst.scene_limits is None
        expected = np.array([0.11076234, 0.0761265, 1.91541779])
        if instance_type == 'periodic':
            expected = expected + periodic_shift
        assert_nested_allclose(inst.center, expected, **tolerances)
        # Altitude from pressure
        kws = instance_kws.copy()
        kws['pressure'] = 2000.0
        kws.pop('altitude', None)
        self.cls(**kws).altitude
        # Specified intensity
        kws = instance_kws.copy()
        kws['intensity'] = 1.0
        self.cls(**kws).intensity_density

    def test_get_solar_direction(self, instance, instance_kws,
                                 skip_if_not_base,
                                 assert_nested_allclose, tolerances):
        expected = instance.forward
        actual = self.cls.get_solar_direction(
            instance_kws['latitude'],
            instance_kws['longitude'],
            instance_kws['date'],
            instance.zenith_direction,
            instance_kws['north'],
            altitude=instance.altitude,
        )
        assert_nested_allclose(actual, expected, **tolerances)

    def test_solar_rotation(self, instance, skip_if_not_base,
                            assert_nested_allclose, tolerances):
        test_point = instance.ground + 100 * instance.zenith_direction
        expected = instance.ground - 100 * instance.forward
        actual = instance.solar_rotation(test_point)
        assert_nested_allclose(actual, expected, **tolerances)
