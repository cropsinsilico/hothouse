import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose
from hothouse import scene


class TestScene:
    r"""Tests for Scene class."""

    cls = scene.Scene
    _instance_kws = {
        'ground': np.array([0.0, 0.0, -19.73708], "f4"),
    }
    _expected_results = {
        'solar_ppfd': np.float32(307746.75),
        'flux_density': np.float32(0.13100827),
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
    def expected_results(self):
        return self._expected_results

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

    def test_attributes(self, instance, nface):
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
            atol=1e-7, rtol=1e-6
        )

    def test_compute_hit_count(self, instance, blaster, nface):
        r"""Test compute_hit_count method."""
        result = instance.compute_hit_count(blaster)
        assert len(result) == 1
        assert result[0].shape == (nface, )
        assert result[0].sum() == blaster.nx * blaster.ny

    def test_compute_solar_ppfd(self, instance, nface, location_champaign,
                                datetime_champaign, expected_results):
        r"""Test compute_solar_ppfd method."""
        result = instance.compute_solar_ppfd(*location_champaign,
                                             datetime_champaign("noon"))
        assert len(result) == 1
        assert result[0].shape == (nface, )
        print(self, result[0].sum())
        assert_allclose(result[0].sum(), expected_results['solar_ppfd'],
                        atol=1e-7, rtol=1e-6)

    def test_compute_flux_density(self, instance, nface, blaster,
                                  expected_results):
        r"""Test compute_flux_density method."""
        result = instance.compute_flux_density(blaster)
        assert len(result) == 1
        assert result[0].shape == (nface, )
        assert_allclose(result[0].sum(), expected_results['flux_density'],
                        atol=1e-7, rtol=1e-6)


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
    def expected_results(self):
        out = copy.deepcopy(self._expected_results)
        out.update(
            solar_ppfd=np.float32(496883.7),
            flux_density=np.float32(0.13121203),
        )
        return out

    def test_periodic_attributes(self, instance, nface):
        r"""Test various periodic attributes."""
        assert instance.nperiodic_copies == 8
        assert instance.ncomponents_periodic == 8
        assert len(instance.transmittance_periodic) == 9
        assert len(instance.reflectance_periodic) == 9
