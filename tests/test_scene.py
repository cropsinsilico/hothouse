import pytest
import copy
import numpy as np
from hothouse import scene


class TestScene:
    r"""Tests for Scene class."""

    cls = scene.Scene
    _instance_kws = {
        'ground': np.array([0.0, 0.0, -19.73708], "f4"),
    }
    _expected_results = {
        'solar_ppfd': np.array([
            3.0774675e5, 1.2462395, 1.5996583e3, 3.2056952e2
        ], "f4"),
        'flux': np.array([
            0.95618385, 0, 0.037574425, 0.0009960248], "f4"),
        'flux_density': np.array([
            0.13100827, 0, 0.0050536012, 0.00013646694], "f4"),
        'tfar': np.array([np.inf, 0.52852875, 1e37, np.inf], "f4"),
    }

    @pytest.fixture(scope="class")
    def instance_kws(self):
        return self._instance_kws

    @pytest.fixture(scope="class")
    def instance(self, instance_kws, model_sphere):
        out = self.cls(**instance_kws)
        out.add_component(model_sphere)
        return out

    @pytest.fixture(scope="class")
    def expected_results(self, blaster):
        out = dict(
            self._expected_results,
            count=np.array([
                blaster.nx * blaster.ny, 0, 9945, 273.066667
            ], "f4"),
        )
        return out

    @pytest.fixture(scope="class")
    def nface(self, model_sphere):
        return model_sphere.triangles.shape[0]

    @pytest.fixture(scope="class")
    def blaster(self, instance):
        from hothouse import blaster
        out = blaster.OrthographicRayBlaster(
            forward=-instance.up,
            center=(instance.ground + 40 * instance.up),
            up=instance.north,
            width=13.956223108821382,  # All rays will hit
            height=13.956223108821382,
        )
        return out

    def test_attributes(self, instance, nface, assert_allclose):
        r"""Test various attributes."""
        assert instance.ncomponents == 1
        assert len(instance.transmittance) == 1
        assert len(instance.reflectance) == 1
        assert instance.transmittance[0].shape == (nface, )
        assert instance.reflectance[0].shape == (nface, )
        assert_allclose(
            instance.limits,
            np.array([
                [-19.73708, -19.73708, -19.73708],
                [-19.73708, -19.73708, 19.73708],
                [19.73708, -19.73708, -19.73708],
                [19.73708, -19.73708, 19.73708],
                [-19.73708, 19.73708, -19.73708],
                [-19.73708, 19.73708, 19.73708],
                [19.73708, 19.73708, -19.73708],
                [19.73708, 19.73708, 19.73708]
            ], "f4"),
        )

    def test_compute_hit_count(self, instance, blaster, nface,
                               expected_results):
        r"""Test compute_hit_count method."""
        result = instance.compute_hit_count(blaster)
        assert len(result) == 1
        assert result[0].shape == (nface, )
        assert result[0].sum() == expected_results['count'][0]

    def test_compute_flux(self, instance, blaster, nface,
                          expected_results):
        r"""Test compute_flux method."""
        result = instance.compute_flux(blaster)
        assert len(result) == 1
        assert result[0].shape == (nface, )
        assert result[0].sum() == expected_results['flux'][0]

    def test_compute_flux_density(self, instance, nface, blaster,
                                  expected_results, assert_allclose):
        r"""Test compute_flux_density method."""
        result = instance.compute_flux_density(blaster)
        assert len(result) == 1
        assert result[0].shape == (nface, )
        assert_allclose(result[0].sum(), expected_results['flux_density'][0])

    def test_compute_count(self, instance, blaster, nface,
                           expected_results, assert_allclose):
        result = instance.compute_count(
            blaster, accumulators={'flux_density': True})
        for k, kresult in result.items():
            assert len(kresult) == 1
            assert kresult[0].shape == (nface, )
            actual = np.array([
                kresult[0].sum(), kresult[0].min(), kresult[0].max(),
                kresult[0].mean()
            ], "f4")
            assert_allclose(actual, expected_results[k])

    def test_compute_solar_ppfd(self, instance, nface, location_champaign,
                                datetime_champaign, expected_results,
                                assert_allclose):
        r"""Test compute_solar_ppfd method."""
        result = instance.compute_solar_ppfd(*location_champaign,
                                             datetime_champaign("noon"))
        assert len(result) == 1
        assert result[0].shape == (nface, )
        actual = np.array([
            result[0].sum(), result[0].min(), result[0].max(),
            result[0].mean()
        ], "f4")
        assert_allclose(actual, expected_results['solar_ppfd'])


class TestPeriodicScene(TestScene):
    r"""Tests for PeriodicScene class."""

    cls = scene.PeriodicScene

    @pytest.fixture(scope="class")
    def instance_kws(self):
        out = copy.deepcopy(self._instance_kws)
        out['period'] = np.array([10.0, 10.0, 0.0], "f4")
        out['count'] = np.array([1, 1, 0], "i4")
        out['buffer_as_primary'] = True
        return out

    @pytest.fixture(scope="class")
    def expected_results(self, blaster):
        out = dict(
            self._expected_results,
            count=np.array([
                blaster.nx * blaster.ny, 0, 18224, 273.066667
            ], "f4"),
            flux=np.array([
                0.9650246, 0, 0.068854332, 0.001005234
            ], "f4"),
            flux_density=np.array([
                0.13121204, 0, 0.0093794512, 0.00013667921
            ], "f4"),
            solar_ppfd=np.array([
                4.9688369e5, 1.2462395, 1.1813491e4, 5.1758716e2
            ], "f4"),
        )
        return out

    def test_periodic_attributes(self, instance, nface):
        r"""Test various periodic attributes."""
        assert instance.nperiodic_copies == 8
        assert instance.ncomponents_periodic == 8
        assert len(instance.transmittance_periodic) == 9
        assert len(instance.reflectance_periodic) == 9
