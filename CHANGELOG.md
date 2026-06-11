# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v0.1.1](https://github.com/cropsinsilico/hothouse/releases/tag/v0.1.1) - 2026-06-11

<small>[Compare with v0.1.0](https://github.com/cropsinsilico/hothouse/compare/v0.1.0...v0.1.1)</small>

## [v0.1.0](https://github.com/cropsinsilico/hothouse/releases/tag/v0.1.0) - 2026-05-27

<small>[Compare with first commit](https://github.com/cropsinsilico/hothouse/compare/9445c3bdc2643cc0e8adcef24614a86ec73843fb...v0.1.0)</small>

### Added

- Add tests to achieve 99% coverage (just missing case of None in check_dtype) Comment out unused and unfinished code Add a more verbose test fixture for checking approximate equality between nested structures (dicts/lists) Add traitlets handlers & decorators for creating dependent properties and updates to traits with dependent defaults ([ad7dd68](https://github.com/cropsinsilico/hothouse/commit/ad7dd6829682e3c8a788b6676e0aac29e69b3cba) by Meagan Lang).
- Add CastAccumulator class for accumulating hit data from the raytracer and apply it to flux, flux_density, distance, & count Update how diffuse intensity is calculate to avoid unnecessary trig methods ([f833be8](https://github.com/cropsinsilico/hothouse/commit/f833be87460de4a85a4760a9c0478e4dffd790aa) by Meagan Lang).
- Add trig functions to mitigate numeric instability Update test fixtures for checking arrays are approximately equal Use a higher tolerance when checking results for multibounce ray tracers Allow multibounce to be specified at the method level ([21845d3](https://github.com/cropsinsilico/hothouse/commit/21845d3d652af1ad65b131e87eef59fb3fc10b5e) by Meagan Lang).
- Additional bug fix for periodic scene and remove use of pkg_resources in setup ([3b491af](https://github.com/cropsinsilico/hothouse/commit/3b491afa8685909990e28b7fb7980e5206a4826d) by Meagan Lang).
- Added projection ray blaster & options for periodic blaster/scene ([3002aaf](https://github.com/cropsinsilico/hothouse/commit/3002aaf50f4b4b12151eee2c45a54974a63f42e3) by Meagan Lang).
- Add method for animating solar illumination between date/times. ([a1728b1](https://github.com/cropsinsilico/hothouse/commit/a1728b1c9b37ab76e72a5f4ca844eba5f24b88c6) by Meagan Lang).
- Add single-bounce option to CLI. ([5b2beda](https://github.com/cropsinsilico/hothouse/commit/5b2beda57788e3454510bbd406b333a31eb8b0b7) by Meagan Lang).
- Adding stubbed callback handler ([7fc3d81](https://github.com/cropsinsilico/hothouse/commit/7fc3d81cb8d18300d440edd83b699a3f3a9f053f) by Matthew Turk).
- adding numpy include ([d97ef6d](https://github.com/cropsinsilico/hothouse/commit/d97ef6dd532282abb98cbb366b00d6125ffe1759) by Matthew Turk).
- Adding new cython file for scene callbacks ([62fcea3](https://github.com/cropsinsilico/hothouse/commit/62fcea30f51118133ea367fa589121263be89b3b) by Matthew Turk).
- Add github actions (#2) ([6b201b6](https://github.com/cropsinsilico/hothouse/commit/6b201b65df7cdaaf6535db988ac765a4ad3218ac) by Matthew Turk).
- Add simple geometry test data. Don't always expect colors to be present in the ply. Ensure that indices read from ply are passed as i4. Update sun_cast script to allow command line options and support other simple geometry shapes. ([d337920](https://github.com/cropsinsilico/hothouse/commit/d337920fb9e6a4efb537f29619f3be0300872a56) by Meagan Lang).
- Add diffuse radiation from pvi to compute_flux_density, allow diffuse radiation to be passed to blasters, update sun_cast to remove unused variables and allow iteration over times. ([09c67d8](https://github.com/cropsinsilico/hothouse/commit/09c67d87e50f939a1a13c938b236a6eaf39185e9) by Meagan Lang).
- Add test GH actions ([9cd2b33](https://github.com/cropsinsilico/hothouse/commit/9cd2b33d93adebd9ca13db1013f9a189ded85c5b) by Matthew Turk).
- adding consistency check test ([c0adb51](https://github.com/cropsinsilico/hothouse/commit/c0adb51f3a6642eb894cd09c63b189c6a243b291) by Matthew Turk).
- add functools.cached_property fallback ([12be842](https://github.com/cropsinsilico/hothouse/commit/12be84231c5a9b6682a142318dab2d035a00e9a1) by Matthew Turk).
- Add intensity as an optional input to ray blasters. Create generic compute_flux_density methods for scene and blaster that compute flux on scene elements or at ray intercepts for a set of light sources represented by ray blasters. Update sun_cast.py to use new format and plot camera image when using blaster. ([f00c5e3](https://github.com/cropsinsilico/hothouse/commit/f00c5e326005ccd251ef5d24f8384ec3be9d1a5e) by Meagan Lang).
- Added ground, up, and north parameters to scene w/ defaults. Added method for getting a SunRayBlaster instance that is guaranteed not to intercept any components in a scene. Added method compute_solar_ppfd for calculating the PPFD based on the suns location for a lat, long, date combination. Remove pysolar references and dependency, replaced with pvlib. Add cached properties to models that stores the norms and areas for triangles. Add ‘model’ script to call pvlib and get the direct/diffuse ppfd based on lat, long and date. Update sun_cast example to make use of compute_solar_ppfd. ([6abfd6a](https://github.com/cropsinsilico/hothouse/commit/6abfd6a7f1f961bd98e98b2a3520cad1493d8b6a) by Meagan Lang).
- Add dedicated blaster class for sun that subclasses OrthographicRayBlaster but uses lat, long, and date to determine where blaster should go to replicate the sun. Add example that uses the new SunRayBlaster. Remove reference to quaternian. Change rotation matrix to be 32bit float to match parameters. ([f228cd1](https://github.com/cropsinsilico/hothouse/commit/f228cd1931907f1bc776aa76581d74ccd4ed7b47) by Meagan Lang).
- Add temp file to ignored files. Add functions for rotation & tests for those functions. ([643aa8a](https://github.com/cropsinsilico/hothouse/commit/643aa8accebb42a70c6da4e8630a1fc6653d2f89) by Meagan Lang).
- Adding a hit-counter ([9af0d02](https://github.com/cropsinsilico/hothouse/commit/9af0d02b3df2110fa6643f781b801daf39322455) by Matthew Turk).
- adding pre-commit ([abf2173](https://github.com/cropsinsilico/hothouse/commit/abf2173fba0caa8f71972769a1644ce780f5f10c) by Matthew Turk).
- Add sun related calculations (#1) ([c13bb2b](https://github.com/cropsinsilico/hothouse/commit/c13bb2b7d3ac9d4f37a68134c318b1320d79a1d4) by cheinemann).
- add axial rotation ([d19ad4e](https://github.com/cropsinsilico/hothouse/commit/d19ad4e3288530bc8c084fd035b47f2cb9c9d4ea) by Matthew Turk).
- Add verbose_output option for cast_once ([8854e01](https://github.com/cropsinsilico/hothouse/commit/8854e010cdb6bf3deaffb77f88cd193ba37c498f) by Matthew Turk).
- Adding initial yggdrasil spec ([0d2e82f](https://github.com/cropsinsilico/hothouse/commit/0d2e82f2953829e56f68c6931534c6f90a7fc5c2) by Matthew Turk).
- Adding imports and an example ([0936461](https://github.com/cropsinsilico/hothouse/commit/093646104ebd6cd960616a404d4bc98b3ad9d04a) by Matthew Turk).
- Adding re-centering and cloning ([1102464](https://github.com/cropsinsilico/hothouse/commit/1102464dc2a7b163acfc4a2d1c88196ae38a80be) by Matthew Turk).
- Adding first test of loading soy ([f90f11d](https://github.com/cropsinsilico/hothouse/commit/f90f11d233b0de085626d37a8434352d669badb8) by Matthew Turk).
- Adding pooch ([85a973e](https://github.com/cropsinsilico/hothouse/commit/85a973e7a6e27d64210e3a0e8ebfdcfc77fc87e8) by Matthew Turk).
- Adding versioneer ([87ff148](https://github.com/cropsinsilico/hothouse/commit/87ff148a2865032872f7c469ea358d9c7712dc8c) by Matthew Turk).

### Fixed

- Fix permissions on docs deployment workflow ([85ed5be](https://github.com/cropsinsilico/hothouse/commit/85ed5be0b5c30bc8f57dda172e97abcc0cd64071) by Meagan Lang).
- Fix warnings in tests ([99d4150](https://github.com/cropsinsilico/hothouse/commit/99d41505ef3a49b9dd101e7872401edc5fbebf89) by Meagan Lang).
- Fix bug where pressure/temperature were not passed to get_solarposition Lower number of decimal places for floating point comparison in scene tests ([18de499](https://github.com/cropsinsilico/hothouse/commit/18de4994cf6b96909f5e91ad9d3225d7e36addab) by Meagan Lang).
- Fix types in blaster Re-arange test scene geometry to make more definiteive hits that will not change due to rounding error ([da73329](https://github.com/cropsinsilico/hothouse/commit/da733294c76362b5fd85c0cfeabfdbb64ff8055b) by Meagan Lang).
- Fix bug in exclusion of periodic scene components ([8781cd8](https://github.com/cropsinsilico/hothouse/commit/8781cd824e5b17e3e17f4edbb1821f506df55fa0) by Meagan Lang).
- Fix bug in rotation of redirected ray and ensure that diffuse radiation is only added when diffuse_intensity is not 0 ([293215e](https://github.com/cropsinsilico/hothouse/commit/293215e900b0f95faddf81b67db92236c5fa2cfa) by Meagan Lang).
- Fix camera for sphere ([5da7d6d](https://github.com/cropsinsilico/hothouse/commit/5da7d6d13b921439ea834544197e0b600518b5ff) by Meagan Lang).
- Fix the origin broadcasting ([be25ea0](https://github.com/cropsinsilico/hothouse/commit/be25ea0f50192f82d1b8c6d43221e99b8347b7c4) by Matthew Turk).

### Changed

- Change to using double precisions for all calcs except the call to embree scene (triangles) and run (origins/directions) ([37427fe](https://github.com/cropsinsilico/hothouse/commit/37427fed896ebe0f59bb7ac0707fcd21110444af) by Meagan Lang).
- Change to using numpy.testing.assert_allclose for testing w/ explicit absolute/relative tolerances ([627be13](https://github.com/cropsinsilico/hothouse/commit/627be135182b8638d58a705fa9b468f01ecb715a) by Meagan Lang).
- Change ray to pass-by-ref in callback and add class for multi-bounce collision that gathers information for each bounce. ([033864e](https://github.com/cropsinsilico/hothouse/commit/033864ee9f078ed5f1e0cca9d0bb2942b771d375) by Meagan Lang).
- Change property from norm to normals to be more descriptive and match obj name, fix error where any_direction==False should have zeroed contribution from face with normal opposite light, and pass tilt to pvlib in degrees, not radians. ([7f4f78f](https://github.com/cropsinsilico/hothouse/commit/7f4f78f749b69f3d803f24b786856ee6d4878fcf) by Meagan Lang).
- Change to CFloat ([d39766b](https://github.com/cropsinsilico/hothouse/commit/d39766bcba3946ed8f9264f82d044aab821c0436) by Meagan Lang).

### Removed

- Remove quaternion dependency and references. Add solarpy & pythreejs as dependencies. ([017ebe4](https://github.com/cropsinsilico/hothouse/commit/017ebe41e11ba32143749946a9e2a8429e205b85) by Meagan Lang).

