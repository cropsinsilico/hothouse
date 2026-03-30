from enum import Enum
import numpy as np
import functools

import traitlets
import traittypes
import datetime
import pvlib

from .traits_support import check_dtype, check_shape

from hothouse import sun_calc
# from .ray_callbacks import RayCollisionMultiBounce
# RayCollisionPrinter

cached_property = getattr(functools, "cached_property", property)

# pyembree receives origins and directions.


class QueryType(Enum):
    r"""Identifier for query that should be run."""
    DISTANCE = "DISTANCE"
    OCCLUDED = "OCCLUDED"
    INTERSECT = "INTERSECT"


class MultibounceCallback(traitlets.HasTraits):
    r"""Callback for completing ray reflectance/transmittance.

    Args:
        origins (np.ndarray): Ray origins.
        directions (np.ndarray): Unit vectors describing ray directions.
        hits (dict): Raytracer results for the provided rays.
        power (np.ndarray, optional): Current power of the rays.
        power_threshold (float, optional): Threshold below which rays
            should no longer be tracked.
        transmittance (list, optional): Arrays of transmittance values
            for each face in the scene components.
        reflectance (list, optional): Arrays reflectance values for
            each face in the scene components.

    """
    # TODO: Multispectral bounces

    origins = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f4"))
    directions = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f4"))
    hits = traitlets.Dict()
    power = traittypes.Array().valid(check_dtype("f4"))
    power_threshold = traitlets.CFloat(0.001)
    original_ray = traittypes.Array().valid(check_dtype("i4"))
    transmittance = traitlets.List()
    reflectance = traitlets.List()

    @traitlets.default("power")
    def _default_power(self):
        return np.ones((self.origins.shape[0], ), "f4")

    @traitlets.default("original_ray")
    def _default_original_ray(self):
        return np.arange(self.origins.shape[0], dtype="i4")

    @cached_property
    def idx(self):
        r"""np.ndarray: Boolean mask for rays with hits."""
        return np.logical_and(
            self.hits['primID'] >= 0,
            self.power >= self.power_threshold
        )

    @cached_property
    def N(self):
        r"""Number of rays with hits."""
        return np.sum(self.idx)

    @cached_property
    def intersections(self):
        r"""np.ndarray: Coordinates of ray intersections."""
        return (
            self.origins[self.idx]
            + sun_calc.op_along_axis(np.multiply,
                                     self.directions[self.idx],
                                     self.hits['tfar'][self.idx])
        )

    @cached_property
    def reflected_directions(self):
        r"""np.ndarray: Unit vectors for the ray reflections."""
        # Rotate inverse of original direction 180 deg around surface
        # normal to get new direction
        return sun_calc.rotate_u(-self.directions[self.idx], np.pi,
                                 self.hits['Ng'][self.idx]).astype("f4")

    @cached_property
    def transmitted_directions(self):
        r"""np.ndarray: Unit vectors for the ray tranmissions."""
        # TODO: Refraction
        return self.directions[self.idx]

    @cached_property
    def reflected_power(self):
        r"""np.ndarray: Power of reflected rays."""
        R = np.zeros((self.N, ), 'float32')
        if self.reflectance:
            for i, (geomID, primID) in enumerate(
                    zip(self.hits['geomID'][self.idx],
                        self.hits['primID'][self.idx])):
                R[i] = self.reflectance[geomID][primID]
        out = self.power[self.idx].copy()
        out[:] *= R
        return out

    @cached_property
    def transmitted_power(self):
        r"""np.ndarray: Power of transmitted rays."""
        T = np.zeros((self.N, ), 'float32')
        if self.transmittance:
            for i, (geomID, primID) in enumerate(
                    zip(self.hits['geomID'][self.idx],
                        self.hits['primID'][self.idx])):
                T[i] = self.transmittance[geomID][primID]
        out = self.power[self.idx].copy()
        out[:] *= T
        return out

    @cached_property
    def next_power(self):
        r"""np.ndarray: Power for the next round of rays."""
        return np.hstack([
            self.reflected_power,
            self.transmitted_power,
        ])

    @cached_property
    def next_original_ray(self):
        r"""np.ndarray: Index of the original ray producing each of
        the rays in the next round."""
        return np.hstack([
            self.original_ray[self.idx],
            self.original_ray[self.idx],
        ])

    @cached_property
    def next_idx(self):
        r"""np.ndarray: Mask for hits that produce rays."""
        return (self.next_power > 0)

    @cached_property
    def next_N(self):
        r"""int: Number of rays resulting from reflection & tramission."""
        if self.N == 0:
            return 0
        return np.sum(self.next_idx)

    @cached_property
    def next_origins(self):
        r"""np.ndarray: Origins for the next round of rays."""
        # Prevent intersection with origin by adding small offset
        out = np.vstack([
            self.intersections,
            self.intersections,
        ])
        out[:] += 1e-4 * self.next_directions
        return out

    @cached_property
    def next_directions(self):
        r"""np.ndarray: Directions for the next round of rays."""
        return np.vstack([
            self.reflected_directions,
            self.transmitted_directions,
        ])

    def next_callback(self, hits):
        r"""Create a callback for the next bounce based on raytrace
        results for this bounce.

        Args:
            hits (dict): Result from this bounce.

        Returns:
            MultibounceCallback: Callback for next bounce.

        """
        assert len(hits['geomID']) == self.next_N
        return MultibounceCallback(
            origins=self.next_origins[self.next_idx, :],
            directions=self.next_directions[self.next_idx, :],
            hits=hits,
            power=self.next_power[self.next_idx],
            original_ray=self.next_original_ray[self.next_idx],
            power_threshold=self.power_threshold,
            transmittance=self.transmittance,
            reflectance=self.reflectance,
        )

    def join_bounces(self, bounces):
        r"""Combine output from a set of bounces.

        Args:
            bounces (list): Output from raytrace for each bounce.

        Returns:
            dict: Combined data from the bounces.

        """
        maxbounce = len(bounces)
        nray = self.origins.shape[0]
        nbounce = np.zeros(nray, dtype="int32")
        for x in bounces:
            idx, idx_count = np.unique(x['original_ray'],
                                       return_counts=True)
            nbounce[idx] += idx_count
        maxbounce = max(nbounce)
        out = {
            'nbounce': np.zeros(nray, dtype="int32"),
            'primID': -1 * np.ones((nray, maxbounce), dtype="int32"),
            'geomID': -1 * np.ones((nray, maxbounce), dtype="int32"),
            'tfar': 1e37 * np.ones((nray, maxbounce), dtype="float32"),
            'u': np.zeros((nray, maxbounce), dtype="float32"),
            'v': np.zeros((nray, maxbounce), dtype="float32"),
            'Ng': np.zeros((nray, maxbounce, 3), dtype="float32"),
            'ray_dir': np.zeros((nray, maxbounce, 3), dtype="float32"),
            'power': np.zeros((nray, maxbounce), dtype="float32"),
        }
        for x in bounces:
            idx_ray = x['original_ray']
            idx_bounce = np.zeros(idx_ray.shape, dtype="int32")
            ray_ids, ray_ids_count = np.unique(idx_ray, return_counts=True)
            for ray_id, ray_count in zip(ray_ids, ray_ids_count):
                idx_bounce[idx_ray == ray_id] = (
                    out['nbounce'][ray_id]
                    + np.arange(ray_count, dtype="i4")
                )
                out['nbounce'][ray_id] += ray_count
            for k in out.keys():
                if k == 'nbounce':
                    continue
                out[k][idx_ray, idx_bounce, ...] = x[k]
        assert (out['nbounce'] == nbounce).all()
        return out


class RayBlaster(traitlets.HasTraits):
    r"""Container for a set of rays.

    Args:
        origins (np.ndarray): Ray origins.
        directions (np.ndarray): Unit vectors describing ray directions.
        intensity (float, optional): Total intensity of light in rays.
        diffuse_intensity (float, optional): Diffuse intensity.
        multibounce (bool, optional): If True, rays should be tracked
            through reflections/transmission.
        power_threshold (float, optional): Threshold below which rays
            should no longer be tracked during bounces.

    """
    origins = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f4"))
    directions = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f4"))
    intensity = traitlets.CFloat(1.0)
    diffuse_intensity = traitlets.CFloat(0.0)
    multibounce = traitlets.CBool(False)
    power_threshold = traitlets.CFloat(0.001)

    @property
    def ray_intensity(self):
        r"""float: Intensity of single ray."""
        return self.intensity / self.origins.shape[0]

    def cast_once(self, scene, verbose_output=False,
                  query_type=QueryType.DISTANCE,
                  origins=None, directions=None, dists=None,
                  in_bounce=False):
        r"""Run the embree raytrace once on a scene.

        Args:
            scene (hothouse.scene.Scene): Scene to raytrace.
            verbose_output (bool, optional): If True, output a dictionary
                of all ray properties.
            query_type (QueryType, optional): Raytrace query type
                controlling the output.
            origins (np.ndarray, optional): Alternate ray origins to use.
            directions (np.ndarray, optional): Alternate ray directions
                to use.
            dists (np.ndarray, optional): Array of distances for each
                ray that should be updated.
            in_bounce (bool, optional): If True, the provided rays are
                for a bounce.

        Returns:
            np.ndarray, dict: Ray trace result.

        """
        if origins is None:
            origins = self.origins
        if directions is None:
            directions = self.directions
        multibounce = self.multibounce
        if (not verbose_output) or in_bounce:
            multibounce = False
        # if self.multibounce:
        #     callback_handler = RayCollisionMultiBounce(
        #         self.origins.shape[0], 10,
        #         scene.transmittance_periodic,
        #         scene.reflectance_periodic)
        # else:
        #     callback_handler = None
        output = scene.embree_scene.run(
            origins,
            directions,
            dists=dists,
            query=query_type._value_,
            output=verbose_output,
            # callback_handler=callback_handler
        )
        if isinstance(output, dict):
            # Patch for uninitialized variables
            for k in ['u', 'v']:
                output[k][output['geomID'] == -1] = 0.0
            output['Ng'][output['geomID'] == -1, :] = 0.0
        output = scene.post_cast(query_type, output)
        if not multibounce:
            return output
        # if self.multibounce and isinstance(output, dict):
        #     output['bounces'] = callback_handler.bounces
        bounces = []
        callback0 = MultibounceCallback(
            origins=origins, directions=directions, hits=output,
            power_threshold=self.power_threshold,
            transmittance=scene.transmittance,
            reflectance=scene.reflectance,
        )
        callback = callback0
        while callback.next_N > 0:
            iout = self.cast_once(
                scene, verbose_output=True, query_type=query_type,
                origins=callback.next_origins,
                directions=callback.next_directions,
                in_bounce=True,
            )
            iout.update(
                ray_dir=callback.next_directions,
                power=callback.next_power,
                original_ray=callback.next_original_ray,
            )
            bounces.append(iout)
            callback = callback.next_callback(iout)
        output['bounces'] = callback0.join_bounces(bounces)
        return output

    def compute_distance(self, scene):
        r"""Get the distance that each ray will travel before hitting
        a surface in the scene.

        Args:
            scene (hothouse.scene.Scene): Scene to cast rays on.

        Returns:
            np.ndarray: Travel distances for each ray.

        """
        output = self.cast_once(
            scene, verbose_output=False, query_type=QueryType.DISTANCE
        )
        return output

    def compute_count(self, scene):
        r"""Run the raytracer to determine how each ray will intersect
        the scene.

        Args:
            scene (hothouse.scene.Scene): Scene to cast rays on.

        Returns:
            dict: Raytrace results::

                geomID (np.ndarray): Index of the scene component that
                    each ray intersects. -1 for no intersection.
                primID (np.ndarray): Index of the component face that
                    each ray intersects within the component geometry
                    that contains it. -1 for no intersection.
                tfar (np.ndarray): The distance that each ray traveled
                    before intersecting a surface in the scene.
                Ng (np.ndarray): Normal vector for the surface that each
                    ray intersected.
                u (np.ndarray): Projection of the ray up along the
                    surface that each ray intersected
                    (barycentric u coordinate of hit).
                v (np.ndarray): Projection of the ray east along the
                    surface that each ray intersected
                    (barycentric v coordinate of hit).

        """
        output = self.cast_once(
            scene, verbose_output=True, query_type=QueryType.INTERSECT
        )
        return output

    def compute_flux_density(self, scene, light_sources, any_direction=True):
        r"""Compute the flux density on each scene element touched by
        this blaster from a set of light sources.

        Args:
            scene (hothouse.scene.Scene): Scene to get flux density for.
            light_sources (list): Set of RayBlasters used to determine
                the light incident on scene elements.
            any_direction (bool, optional): If True, the flux is deposited
                on component reguardless of if the blaster ray hits the
                front or back of a component surface. If False, flux
                is only deposited if the blaster ray hits the front.
                Defaults to True.

        Returns:
            array: Total flux density on surfaces intercepted by the
                rays.

        """
        fd_scene = scene.compute_flux_density(
            light_sources, any_direction=any_direction
        )
        out = np.zeros(self.nx * self.ny, "f4")
        camera_hits = self.compute_count(scene)
        for ci, component in enumerate(scene.components):
            idx_ci = np.where(camera_hits["geomID"] == ci)[0]
            hits = camera_hits["primID"][idx_ci]
            out[idx_ci[hits >= 0]] += fd_scene[ci][hits[hits >= 0]]
        return out


class OrthographicRayBlaster(RayBlaster):
    r"""Container for orthographic set of rays in a square.

    Args:
        center (np.ndarray): Center of the set of rays.
        forward (np.ndarray): Unit vector giving the ray direction.
        up (np.ndarray, optional): Unit vector giving the y direction
            of the ray pattern. Required if east not provided.
        east (np.ndarray, optional): Unit vector giving the x direction
            of the ray pattern. Required if up not provided.
        width (float, optional): Width of the ray pattern.
        height (float, optional): Height of the ray pattern.
        nx (int, optional): Number of rays distributed in the x (east)
            direction.
        ny (int, optional): Number of rays distributed in the y (up)
            direction.
        period (np.ndarray, optional): Number of times the rays should be
            repeated in the up & east directions.
        intensity (float, optional): Total intensity of light in rays.
        intensity_density (float, optional): Intensity of light in rays
            per unit area. If provided and intensity is not, this will be
            used to calculate intensity.
        **kwargs: Additional keyword arguments are passed to RayBlaster.

    """

    center = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    forward = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    up = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    east = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    width = traitlets.CFloat(1.0)
    height = traitlets.CFloat(1.0)
    nx = traitlets.CInt(512)
    ny = traitlets.CInt(512)
    period = traittypes.Array(
        np.zeros((3,), "f4")
    ).valid(check_dtype("f4"), check_shape(3))
    intensity = traitlets.CFloat()
    intensity_density = traitlets.CFloat()

    @traitlets.default("intensity")
    def _default_intensity(self):
        if not self.trait_has_value("intensity_density"):
            return 1.0
        return self.intensity_density * self.width * self.height

    @traitlets.default("intensity_density")
    def _default_intensity_density(self):
        return self.intensity / (self.width * self.height)

    @traitlets.default("forward")
    def _default_forward(self):
        if not (self.trait_has_value("up")
                and self.trait_has_value("forward")):
            raise ValueError("At least two direction must be provided "
                             "out of forward, up, & east")
        return np.cross(self.up, self.east)

    @traitlets.default("up")
    def _default_up(self):
        return np.cross(self.east, self.forward)

    @traitlets.default("east")
    def _default_east(self):
        return np.cross(self.forward, self.up)

    @traitlets.default("directions")
    def _default_directions(self):
        self._directions = np.zeros((self.nx, self.ny, 3), dtype="f4")
        self._directions[:] = self.forward[None, None, :]
        return self._directions.view().reshape((self.nx * self.ny, 3))

    @traitlets.default("origins")
    def _default_origins(self):
        # here origin is not the center, but the bottom left
        self._origins = np.zeros((self.nx, self.ny, 3), dtype="f4")
        offset_x, offset_y = np.mgrid[
            (-self.width / 2):(self.width / 2):(self.nx * 1j),
            (-self.height / 2):(self.height / 2):(self.ny * 1j),
        ]
        self._origins[:] = (
            self.center
            + offset_x[..., None] * self.east
            + offset_y[..., None] * self.up
        )
        return self._origins.view().reshape((self.nx * self.ny, 3))

    @cached_property
    def is_periodic(self):
        r"""bool: True if the blaster will be replicated periodically."""
        return np.any(self.period > 0)

    def cast_once(self, scene, **kwargs):
        r"""Run the embree raytrace once on a scene.

        Args:
            scene (hothouse.scene.Scene): Scene to raytrace.
            **kwargs: Additional keyword arguments are passed to the
                RayBlaster method.

        Returns:
            np.ndarray, dict: Ray trace result.

        """
        if ((self.is_periodic and 'dists' not in kwargs
             and not kwargs.get('in_bounce', False))):
            kwargs['dists'] = np.empty(
                kwargs.get('origins', self.origins).shape[0], 'float32')
            kwargs['dists'].fill(1e37)
        out = super(OrthographicRayBlaster, self).cast_once(
            scene, **kwargs
        )
        if self.is_periodic and not kwargs.get('in_bounce', False):
            from hothouse.scene import PeriodicScene
            origins = kwargs.pop('origins', self.origins)
            # TODO: Gap between repetitions?
            shifts = PeriodicScene.get_periodic_shifts(
                np.ndarray([self.width, self.height, 0.0]),
                np.hstack([self.east, self.up, self.forward]).T,
                self.period)
            for shift in shifts:
                # TODO: Merge this?
                out = self.cast_once(
                    scene, origins=(origins + shift), **kwargs
                )
        return out


class SunRayBlaster(OrthographicRayBlaster):
    r"""Container for an orthographic set of rays with properties set
    based on the position of the sun for a given location and date/time.

    Args:
        latitude (float): Location latitude (in degrees).
        longitude (float): Location longitude (in degrees).
        date (datetime.datetime): Date/time.
        ground (np.ndarray): Position of the center of the ray
            projection on the ground.
        zenith (np.ndarray): Position directly above 'ground' at distance
            that sun blaster should be placed.
        north (np.ndarray): Direction of north on ground from 'ground'.
        solar_altitude (float): Angle of sun above the horizon.
        solar_azimuth (float): Angle of sun around the horizon from
            north.
        solar_distance (float): Distance of sun from ground.
        scene_limits (np.ndarray): Set of points defining the bounds of
            the scene.
        altitude (float, optional): Altitude (in meters) used to compute
            solar position. If not provided, but pressure is, pressure
            will be used to calculate altitude.
        pressure (float, optional): Pressure (in Pa) used to compute
            solar position. If not provided, but altitude is, altitude
            will be used to calculate pressure.
        temperature (float, optional): Air temperature (in degrees C)
            used to compute solar position.
        eta_par (float, optional): Fraction of solar radiation (assuming
            black-body spectrum of 5800 K) that is photosynthetically
            active (wavelengths 400–700 nm).
        eta_photon (float, optional): Average number of photons per
            photosynthetically activate unit of radiation (in
            µmol s−1 W−1).

    """

    latitude = traitlets.Float()
    longitude = traitlets.Float()
    date = traitlets.Instance(klass=datetime.datetime)

    ground = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    zenith = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    north = traittypes.Array().valid(check_dtype("f4"), check_shape(3))

    solar_altitude = traitlets.CFloat()
    solar_azimuth = traitlets.CFloat()
    solar_distance = traitlets.CFloat()
    _solpos_info = traittypes.DataFrame()
    scene_limits = traittypes.Array(None, allow_none=True).valid(
        check_shape(None, 3), check_dtype("f4"))

    altitude = traitlets.Float()
    pressure = traitlets.Float()
    temperature = traitlets.Float(12.0)
    eta_par = traitlets.Float(0.368)
    eta_photon = traitlets.Float(4.56)

    @traitlets.default("forward")
    def _default_forward(self):
        # Negative to point from sun to the earth rather than from
        # earth to the sun
        return -self.solar_rotation(self.zenith_direction, is_ray=True)

    @traitlets.default("up")
    def _default_up(self):
        # The "east" used here is not the "east" used elsewhere.
        # This is the east wrt north etc, but we need an east
        # for blasting from elsewhere.
        return -self.solar_rotation(self.solar_east, is_ray=True)

    @traitlets.default("_solpos_info")
    def _default_solpos_info(self):
        return pvlib.solarposition.get_solarposition(
            self.date, self.latitude, self.longitude
        )

    @traitlets.default("solar_altitude")
    def _default_solar_altitude(self):
        solar_altitude = self._solpos_info["apparent_elevation"].iloc[0]
        if solar_altitude < 0:
            raise ValueError(
                f"For the provided lat ({self.latitude}), "
                f"long ({self.longitude}), date, & time "
                f"({self.date}) the sun will be below the horizon."
            )
        return solar_altitude

    @traitlets.default("solar_azimuth")
    def _default_solar_azimuth(self):
        return self._solpos_info["azimuth"].iloc[0]

    @traitlets.default("solar_distance")
    def _default_solar_distance(self):
        zd_nonorm = self.zenith - self.ground
        return np.linalg.norm(zd_nonorm)

    @traitlets.default("width")
    def _default_width(self):
        out = 1.0
        if self.scene_limits is not None:
            out = self.limits_east[1] - self.limits_east[0]
        return out

    @traitlets.default("height")
    def _default_height(self):
        out = 1.0
        if self.scene_limits is not None:
            out = self.limits_up[1] - self.limits_up[0]
        return out

    @traitlets.default("center")
    def _default_center(self):
        if self.scene_limits is not None:
            out = (
                np.mean(self.limits_east) * self.east
                + np.mean(self.limits_up) * self.up
                - self.solar_distance * self.forward
            )
            return out
        v = self.ground - self.solar_distance * self.forward
        offset = max(
            0.0,
            (
                (self.height / 2.0)
                - np.abs(
                    np.linalg.norm(v - self.ground)
                    * np.tan(np.radians(self.solar_altitude))
                )
            ),
        )
        v = v + offset * self.up
        return v

    @traitlets.default("altitude")
    def _default_altitude(self):
        if not self.trait_has_value("pressure"):
            return 0.0
        import pvlib
        return pvlib.atmosphere.pres2alt(self.pressure)

    @traitlets.default("pressure")
    def _default_pressure(self):
        import pvlib
        return pvlib.atmosphere.alt2pres(self.altitude)

    @traitlets.default("intensity_density")
    def _default_intensity_density(self):
        return self.solar_ppfd['direct']

    @traitlets.default("diffuse_intensity")
    def _default_diffuse_intensity(self):
        return self.solar_ppfd['diffuse']

    @traitlets.default("intensity")
    def _default_intensity(self):
        return self.intensity_density * self.width * self.height

    @cached_property
    def solar_ppfd(self):
        r"""dict: Solar intensity calculated based on the location."""
        return sun_calc.solar_ppfd(
            self.latitude, self.longitude, self.date,
            altitude=self.altitude, pressure=self.pressure,
            temperature=self.temperature, eta_par=self.eta_par,
            eta_photon=self.eta_photon,
        )

    @classmethod
    def get_solar_direction(cls, latitude, longitude, date, up, north):
        r"""Get the direction that rays from the sun will travel for
        a given location, time, and orientation.

        Args:
            latitude (float): Location latitude (in degrees).
            longitude (float): Location longitude (in degrees).
            date (datetime.datetime): Date/time.
            up (np.ndarray, optional): Normal unit vector for the ground.
            north (np.ndarray): Unit vector for the north cardinal
                direction.

        Returns:
            SunRayBlaster: Blaster for the location/time.

        """
        instance = cls(latitude=latitude, longitude=longitude,
                       date=date, zenith=(10 * up),
                       ground=np.zeros((3,), "f4"), north=north)
        return instance.forward

    @property
    def zenith_direction(self):
        r"""np.ndarray: Unit vector normal to the ground."""
        zd_nonorm = self.zenith - self.ground
        solar_distance = np.linalg.norm(zd_nonorm)
        return zd_nonorm / solar_distance

    def solar_rotation(self, point, is_ray=False):
        r"""Rotate a point according to same rotation that moves
        sun from the zenith to its location in the sky.

        Args:
            point (array): 3D point to rotate
            is_ray (bool, optional): If True, the point is treated as
                a ray and will not be shifted prior to rotation.

        """

        if is_ray:
            origin = 0.0
        else:
            origin = self.ground
        return sun_calc.rotate_u(
            sun_calc.rotate_u(point - origin,
                              np.radians(90 - self.solar_altitude),
                              self.north),
            np.radians(90 - self.solar_azimuth),
            self.zenith_direction,
        ).astype("f4") + origin

    @cached_property
    def limits_east(self):
        r"""Limits projected in the east direction."""
        adotb = (np.dot(self.scene_limits, self.east)
                 / np.linalg.norm(self.east))
        return [adotb.min(), adotb.max()]

    @cached_property
    def limits_up(self):
        r"""Limits projected in the up direction."""
        adotb = (np.dot(self.scene_limits, self.up)
                 / np.linalg.norm(self.up))
        return [adotb.min(), adotb.max()]

    @property
    def solar_east(self):
        r"""Direction of geographic east."""
        out = np.cross(self.north, self.zenith_direction)
        return out


class ProjectionRayBlaster(RayBlaster):
    r"""Container for a set of rays projected from a camera image plane.

    Args:
        fov_width (float): Field of view width (in degrees).
        fov_height (float): Field of view height (in degrees).
        center (np.ndarray): Center of the set of rays.
        forward (np.ndarray): Unit vector giving the ray direction at the
            center of the field of view.
        up (np.ndarray, optional): Unit vector giving the y direction
            of the ray pattern. Required if east not provided.
        east (np.ndarray, optional): Unit vector giving the x direction
            of the ray pattern. Required if up not provided.
        width (float, optional): Width of the ray pattern.
        height (float, optional): Height of the ray pattern.
        nx (int, optional): Number of rays distributed in the x (east)
            direction.
        ny (int, optional): Number of rays distributed in the y (up)
            direction.
        multibounce (bool, optional): If True, rays should be tracked
            through reflections/transmission.

    """
    fov_width = traitlets.CFloat(90.0)
    fov_height = traitlets.CFloat(90.0)
    center = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    forward = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    up = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    east = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    width = traitlets.CFloat(1.0)
    height = traitlets.CFloat(1.0)
    nx = traitlets.CInt(512)
    ny = traitlets.CInt(512)
    multibounce = traitlets.CBool(False)

    @traitlets.default("east")
    def _default_east(self):
        return np.cross(self.forward, self.up)

    @traitlets.default("directions")
    def _default_directions(self):
        self._directions = self.origins - self.camera_origin
        self._directions = sun_calc.norm_along_axis(
            self._directions, 1).astype("f4")
        return self._directions.view()

    @traitlets.default("origins")
    def _default_origins(self):
        # here origin is not the center, but the bottom left
        self._origins = np.zeros((self.nx, self.ny, 3), dtype="f4")
        offset_x, offset_y = np.mgrid[
            (-self.width / 2):(self.width / 2):(self.nx * 1j),
            (-self.height / 2):(self.height / 2):(self.ny * 1j),
        ]
        self._origins[:] = (
            self.center
            + offset_x[..., None] * self.east
            + offset_y[..., None] * self.up
        )
        return self._origins.view().reshape((self.nx * self.ny, 3))

    @cached_property
    def camera_distance(self):
        r"""float: Distance of the camera from the image plane."""
        return (
            (self.width / 2.0)
            / np.tan(np.radians(self.fov_width / 2.0))
        )

    @cached_property
    def camera_origin(self):
        r"""np.ndarray: Position of the camera."""
        return self.center - self.camera_distance * self.forward


class SphericalRayBlaster(ProjectionRayBlaster):
    r"""Container for a set of spherically distributed rays.

    Args:
        fov_width (float): Field of view width (in degrees). This will be
            the angle of rotation of the rays around the forward
            direction.
        fov_height (float): Field of view height (in degrees). This will
            be the angle of rotation of rays away from the forward
            direction.
        center (np.ndarray): Center of the set of rays.
        forward (np.ndarray): Unit vector giving the ray direction at the
            center of the field of view.
        dont_include_center (bool, optional): If True, don\'t include an
            explicit ray along the forward vector. If False, the center
            ray will be the first ray.

    """
    fov_width = traitlets.CFloat(360.0)
    fov_height = traitlets.CFloat(180.0)
    width = None
    height = None
    camera_distance = None
    camera_origin = None
    dont_include_center = traitlets.CBool(False)

    @traitlets.default("directions")
    def _default_directions(self):
        self._directions = np.zeros((self.nx * self.ny, 3), dtype="f4")
        self._directions[:] = self.forward[None, :]
        fov_width = self.fov_width
        fov_height = self.fov_height
        offset_x, offset_y = np.mgrid[
            (-fov_width / 2):(fov_width / 2):(self.nx * 1j),
            (fov_height / self.ny):fov_height:(self.ny * 1j),
        ]
        offset_x = np.radians(offset_x.flatten())
        offset_y = np.radians(offset_y.flatten())
        self._directions = sun_calc.rotate_u(
            self._directions, -offset_y, self.east)
        self._directions = sun_calc.rotate_u(
            self._directions, offset_x, self.forward)
        self._directions = sun_calc.norm_along_axis(self._directions, 1)
        if not self.dont_include_center:
            self._directions = np.vstack([self.forward, self._directions])
        self._directions = self._directions.astype("f4")
        return self._directions.view()

    @traitlets.default("origins")
    def _default_origins(self):
        # here origin is not the center, but the bottom left
        if self.dont_include_center:
            self._origins = np.zeros((self.nx, self.ny, 3), dtype="f4")
            self._origins[:] = self.center[None, None, :]
            return self._origins.view().reshape((self.nx * self.ny, 3))
        self._origins = np.zeros((self.nx * self.ny + 1, 3), dtype="f4")
        self._origins[:] = self.center[None, None, :]
        return self._origins.view()
