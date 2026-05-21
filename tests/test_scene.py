import pytest
import copy
import numpy as np
import traitlets
from hothouse import scene


class TestScene:
    r"""Tests for Scene class."""

    cls = scene.Scene
    _instance_kws = {
        'ground': np.array([0.0, 0.0, -19.73708], "f8"),
    }
    _expected_results = {
        'solar_ppfd': np.array([
            3.0774675e5, 1.246241953, 1.5996583e3, 3.2056952e2
        ], "f8"),
        'flux': np.array([
            0.95618385, 0, 0.037574425, 0.0009960248], "f8"),
        'flux_density': np.array([
            0.13100827, 0, 0.0050536018, 0.00013646694], "f8"),
        'tfar': np.array([
            33.55874, 0.52852875, 2.3789017, 1.0487106], "f4"),
    }

    @pytest.fixture(scope="class")
    def instance_type(self):
        return "base"

    @pytest.fixture(scope="class")
    def skip_if_not_base(self, instance_type):
        if instance_type != 'base':
            pytest.skip("Only enabled for base instance")

    @pytest.fixture(scope="class")
    def tolerances(self):
        return {}

    @pytest.fixture(scope="class")
    def instance_kws(self, instance_type):
        if instance_type == "base":
            return self._instance_kws
        raise NotImplementedError(instance_type)

    @pytest.fixture(scope="class", params=["sphere"])
    def instance_geometry(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def instance_model(self, geometry_model, instance_geometry):
        return geometry_model(instance_geometry)

    @pytest.fixture(scope="class")
    def instance(self, instance_kws, instance_model):
        out = self.cls(**instance_kws)
        out.add_component(instance_model)
        return out

    @pytest.fixture(scope="class")
    def expected_results(self, blaster):
        out = dict(
            self._expected_results,
            count=np.array([
                blaster.nx * blaster.ny, 0, 9945, 273.066667
            ], "f8"),
        )
        return out

    @pytest.fixture(scope="class")
    def nface(self, instance_model):
        return instance_model.triangles.shape[0]

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

    def test_traits_errors(self, skip_if_not_base):
        r"""Test errors raised for invalid traits."""
        with pytest.raises(traitlets.TraitError):
            self.cls(ground=np.zeros((2, )))
        with pytest.raises(traitlets.TraitError):
            self.cls(ground=np.zeros((2, 3)))
        with pytest.raises(traitlets.TraitError):
            self.cls(ground=np.zeros((3, ), "i4"))

    def test_attributes(self, instance, nface, assert_allclose,
                        skip_if_not_base):
        r"""Test various attributes."""
        assert instance.ncomponents == 1
        assert len(instance.components[0].attributes) == 2
        assert (instance.components[0].attributes['transmittance'].shape
                == (nface, ))
        assert (instance.components[0].attributes['reflectance'].shape
                == (nface, ))
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
            ], "f8"),
        )

    def test_compute_hit_count(self, instance, blaster, nface,
                               expected_results):
        r"""Test compute_hit_count method."""
        result = instance.compute_hit_count(blaster)
        assert len(result) == 1
        assert result[0].shape == (nface, )
        assert result[0].sum() == expected_results['count'][0]

    def test_compute_flux(self, instance, blaster, nface,
                          expected_results, assert_allclose, tolerances):
        r"""Test compute_flux method."""
        result = instance.compute_flux(blaster)
        assert len(result) == 1
        assert result[0].shape == (nface, )
        assert_allclose(result[0].sum(), expected_results['flux'][0],
                        **tolerances)

    def test_compute_flux_density(self, instance, nface, blaster,
                                  expected_results, assert_allclose,
                                  tolerances):
        r"""Test compute_flux_density method."""
        result = instance.compute_flux_density(blaster)
        assert len(result) == 1
        assert result[0].shape == (nface, )
        assert_allclose(result[0].sum(),
                        expected_results['flux_density'][0],
                        **tolerances)

    def test_compute_count(self, instance, blaster, nface,
                           expected_results,
                           assert_nested_allclose, tolerances):
        result = instance.compute_count(blaster)
        actual = {}
        for k, kresult in result.items():
            assert len(kresult) == 1
            assert kresult[0].shape == (nface, )
            if k == 'tfar':
                idx = (kresult[0] != 1e37)
                actual[k] = np.array([
                    kresult[0][idx].sum(), kresult[0][idx].min(),
                    kresult[0][idx].max(),
                    kresult[0][idx].mean()
                ], "f4")
            else:
                actual[k] = np.array([
                    kresult[0].sum(), kresult[0].min(), kresult[0].max(),
                    kresult[0].mean()
                ], "f8")
        assert_nested_allclose(
            actual, expected_results,
            ignore_keys=['flux_density', 'solar_ppfd'],
            **tolerances
        )

    def test_accumulate_multibounce(self, geometry_scene, blaster,
                                    assert_nested_allclose,
                                    tolerances):
        instance = geometry_scene("pyramid")
        counts = blaster.compute_count(instance, multibounce=True)
        accum = scene.CastAccumulator.from_kwargs(
            'flux', (instance._accumulate_flux, {}))
        res = {}
        accum.accumulate(
            res, 0, instance.components[0], counts,
            ray_dir=blaster.directions,
            ray_intensity=blaster.ray_intensity,
        )
        res2 = instance.compute_count(
            blaster, multibounce=True,
            accumulators={'flux': accum, 'count': False, 'tfar': False},
        )
        assert_nested_allclose(res2, res, **tolerances)

        def dummy_func(dst, *args, **kwargs):
            dst += 1

        res = {}
        accum2 = scene.CastAccumulator.from_kwargs('dummy', dummy_func)
        accum2.accumulate(
            res, 0, instance.components[0], counts,
            ray_dir=blaster.directions,
            ray_intensity=blaster.ray_intensity,
        )
        assert_nested_allclose(
            res, {
                'dummy': {
                    0: 2 * np.ones(instance.components[0].nface),
                },
            },
            **tolerances
        )

    def test_compute_solar_ppfd(self, instance, nface, location_champaign,
                                datetime_champaign, expected_results,
                                assert_allclose, tolerances_solar):
        r"""Test compute_solar_ppfd method."""
        result = instance.compute_solar_ppfd(*location_champaign,
                                             datetime_champaign("noon"))
        assert len(result) == 1
        assert result[0].shape == (nface, )
        actual = np.array([
            result[0].sum(), result[0].min(), result[0].max(),
            result[0].mean()
        ], "f8")
        assert_allclose(actual, expected_results['solar_ppfd'],
                        **tolerances_solar)

    def test_get_sun_blaster(self, instance, location_champaign,
                             datetime_champaign, assert_allclose,
                             tolerances_solar):
        r"""Test get_sun_blaster method."""
        blaster = instance.get_sun_blaster(
            *location_champaign, datetime_champaign("noon"),
            direct_ppfd=0, diffuse_ppfd=0)
        assert (blaster.ray_intensity == 0).all()

    @pytest.mark.slow
    def test_animate_sun(self, tmp_path, geometry_scene, blaster,
                         location_champaign, skip_if_not_base,
                         datetime_champaign, altitude_champaign):
        r"""Test animate_sun method."""
        # Don't use sphere for this test as multibounce takes a long time
        instance = geometry_scene("pyramid")
        anim_dir = tmp_path / "anim"
        anim_dir.mkdir()
        fname = anim_dir / "animate_pyramid.html"
        t_start = datetime_champaign("sunrise")
        t_stop = datetime_champaign("sunset")
        n_step = 3
        anim = instance.animate_sun(
            blaster, *location_champaign,
            t_start, t_stop, n_step,
            altitude=altitude_champaign,
            multibounce=False,
            fname=fname,
        )
        del anim
        import matplotlib.pyplot as plt
        with plt.ion():
            anim = instance.animate_sun(
                blaster, *location_champaign,
                t_start, t_stop, n_step,
                altitude=altitude_champaign,
            )
            del anim

    def test_ipython_display_(self, instance, skip_if_not_base):
        instance._ipython_display_()


@pytest.mark.parametrize("instance_type", ["base", "dont_reflect"],
                         scope="class")
class TestPeriodicScene(TestScene):
    r"""Tests for PeriodicScene class."""

    cls = scene.PeriodicScene
    test_animate_sun = None
    instance_type = None

    @pytest.fixture(scope="class")
    def instance_kws(self, instance_type):
        out = copy.deepcopy(self._instance_kws)
        out['period'] = np.array([10.0, 10.0, 0.0], "f8")
        out['count'] = np.array([1, 1, 0], "i4")
        out['buffer_as_primary'] = True
        if instance_type == 'base':
            return out
        if instance_type == 'dont_reflect':
            out['dont_reflect'] = True
            out['dont_center'] = True
            out['count'] *= 2
            return out
        raise NotImplementedError(instance_type)

    @pytest.fixture(scope="class")
    def expected_results(self, blaster, instance_type):
        out = dict(
            self._expected_results,
            tfar=np.array([
                28.76002, 0.52852875, 1.2916243, 0.95866734
            ], "f4"),
            count=np.array([
                blaster.nx * blaster.ny, 0, 18224, 273.066667
            ], "f8"),
            flux=np.array([
                0.9650246, 0, 0.068854332, 0.001005234
            ], "f8"),
            flux_density=np.array([
                0.13121204, 0, 0.0093794512, 0.00013667921
            ], "f8"),
            solar_ppfd=np.array([
                4.9688369e5, 1.246241953, 1.1813491e4, 5.1758716e2
            ], "f8"),
        )
        if instance_type == 'dont_reflect':
            out['count'][2] = 20306
            out.update(
                tfar=np.array([
                    31.159254, 0.52852875, 2.378777742, 1.00513732,
                ], "f8"),
                flux=np.array([
                    0.9605579, 0, 0.0709626989, 0.0010005812,
                ], "f8"),
                flux_density=np.array([
                    0.131111456, 0, 0, 0,
                ], "f8"),
                solar_ppfd=np.array([
                    4.09237673e5, 1.24624195, 5.39298144e3, 4.26289242e2
                ], "f8"),
            )
        return out

    def test_periodic_attributes(self, instance, nface):
        r"""Test various periodic attributes."""
        if instance.dont_reflect:
            assert instance.nperiodic_copies == 3
            assert instance.ncomponents_periodic == 3
        else:
            assert instance.nperiodic_copies == 8
            assert instance.ncomponents_periodic == 8
