from enum import Enum
import numpy as np
import functools

import traitlets
import traittypes
import datetime
import pvlib

from .traits_support import (
    check_dtype, check_shape, dependent_default, dependent_property
)

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


class RayBlaster(traitlets.HasTraits):
    r"""Container for a set of rays.

    Args:
        origins (np.ndarray): Ray origins.
        directions (np.ndarray): Unit vectors describing ray directions.
        ray_intensity (np.ndarray, optional): Intensity of light in each
           ray.
        intensity (float, optional): Total intensity of light in rays.
        diffuse_intensity (float, optional): Diffuse intensity.

    """
    origins = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f8"))
    directions = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f8"))
    ray_intensity = traittypes.Array().valid(check_dtype("f8"))
    intensity = traitlets.CFloat()
    diffuse_intensity = traitlets.CFloat(0.0)

    _null_field_values = {
        'primID': np.int32(-1),
        'geomID': np.int32(-1),
        'tfar': np.float32(1e37),
    }

    @dependent_default("intensity", ["ray_intensity"])
    def _default_intensity(self):
        if self.trait_has_value("ray_intensity"):
            return self.ray_intensity.sum()
        return 1.0

    @dependent_default("ray_intensity", ["origins", "intensity"])
    def _default_ray_intensity(self):
        out = np.empty((self.origins.shape[0], ), "f8")
        out.fill(self.intensity / len(out))
        return out

    @property
    def nray(self):
        r"""int: Number of rays in the blaster."""
        return self.origins.shape[0]

    @property
    def _check_shape_nray(self):
        return check_shape(self.nray, ignore_trailing=True)

    @traitlets.validate("origins", "directions", "ray_intensity")
    def _validate_nray(self, proposal):
        self._check_shape_nray(proposal['trait'], proposal['value'])
        return proposal['value']

    def cast_once(self, scene, verbose_output=False,
                  query_type=QueryType.DISTANCE, dists=None,
                  multibounce=False, include_attributes=False, **kwargs):
        r"""Run the embree raytrace once on a scene.

        Args:
            scene (hothouse.scene.Scene): Scene to raytrace.
            verbose_output (bool, optional): If True, output a dictionary
                of all ray properties.
            query_type (QueryType, optional): Raytrace query type
                controlling the output.
            dists (np.ndarray, optional): Array of distances for each
                ray that should be updated.
            multibounce (bool, optional): If True, rays should be tracked
                through reflections/transmission.
            include_attributes (bool, list): If True, include all face
                attributes for each intercepted face in the results. If
                a list is provided, just the specified attributes will
                be included. This will be ignored if verbose_output is
                False.
            **kwargs: Additional keyword arguments are passed to
                BouncedRayBlaster if multibounce is True.

        Returns:
            np.ndarray, dict: Ray trace result.

        """
        if (not verbose_output):
            multibounce = False
        if multibounce:
            if include_attributes is True and scene.components:
                include_attributes = list(
                    scene.components[0].attributes.keys())
            if not isinstance(include_attributes, list):
                include_attributes = []
            include_attributes = include_attributes + [
                x for x in ["reflectance", "transmittance"]
                if x not in include_attributes
            ]
        elif not verbose_output:
            include_attributes = False
        # if multibounce:
        #     callback_handler = RayCollisionMultiBounce(
        #         self.nray, 10,
        #         scene.transmittance_periodic,
        #         scene.reflectance_periodic)
        # else:
        #     callback_handler = None
        origins_f4 = self.origins.astype("f4")
        directions_f4 = self.directions.astype("f4")
        output = scene.embree_scene.run(
            origins_f4,
            directions_f4,
            dists=dists,
            query=query_type._value_,
            output=verbose_output,
            # callback_handler=callback_handler
        )
        if isinstance(output, dict):
            # Patch for uninitialized variables
            idx = (output['geomID'] == -1)
            output['Ng'][idx, :] = 0.0
            for k in ['u', 'v']:
                output[k][idx] = 0.0
        output = scene.post_cast(query_type, output)
        if include_attributes:
            if include_attributes is True:
                include_attributes = (
                    list(scene.components[0].attributes.keys())
                    if scene.components else []
                )
            for k in include_attributes:
                assert k not in output
                output[k] = np.zeros(self.nray, "f8")
            for ci, component in enumerate(scene.components):
                idx = (output['geomID'] == ci)
                for k in include_attributes:
                    if k not in component.attributes:
                        continue
                    output[k][idx, ...] = component.attributes[
                        k][output['primID'][idx], ...]
        if not multibounce:
            return output
        # if multibounce and isinstance(output, dict):
        #     output['bounces'] = callback_handler.bounces
        bounces = []
        bounce0 = self.bounce(output, **kwargs)
        bounce = bounce0
        while bounce.nray > 0:
            iout = bounce.cast_once(
                scene, query_type=query_type,
                include_attributes=include_attributes,
            )
            bounces.append(iout)
            bounce = bounce.bounce(iout)
        output['bounces'] = bounce0.join_bounces(bounces)
        return output

    def bounce(self, hits, **kwargs):
        r"""Create a ray tracer for bounced rays.

        Args:
            hits (dict, optional): Results from running this ray tracer.
            **kwargs: Additional keyword arguments are passed to
                BouncedRayBlaster.

        Returns:
            BouncedRayBlaster: Blaster containing ray information for the
                bounced rays.

        """
        for k in ['origins', 'directions', 'ray_intensity']:
            kwargs.setdefault(f'prev_{k}', getattr(self, k))
        return BouncedRayBlaster(hits=hits, **kwargs)

    def compute_occluded(self, scene):
        r"""Get a mask for rays that will be occluded.

        Args:
            scene (hothouse.scene.Scene): Scene to cast rays on.

        Returns:
            np.ndarray: Mask for occluded rays, where 0 indicates an
                occluded ray and -1 indicated unoccluded.

        """
        return self.cast_once(
            scene, verbose_output=False, query_type=QueryType.OCCLUDED,
        )

    def compute_intersect(self, scene):
        r"""Get the indices of intersected faces that the rays intersect.

        Args:
            scene (hothouse.scene.Scene): Scene to cast rays on.

        Returns:
            np.ndarray: Indices of faces that are intersected. -1
                indicates no intersection.

        """
        return self.cast_once(
            scene, verbose_output=False, query_type=QueryType.INTERSECT,
        )

    def compute_distance(self, scene):
        r"""Get the distance that each ray will travel before hitting
        a surface in the scene.

        Args:
            scene (hothouse.scene.Scene): Scene to cast rays on.

        Returns:
            np.ndarray: Travel distances for each ray.

        """
        return self.cast_once(
            scene, verbose_output=False, query_type=QueryType.DISTANCE,
        )

    def compute_count(self, scene, **kwargs):
        r"""Run the raytracer to determine how each ray will intersect
        the scene.

        Args:
            scene (hothouse.scene.Scene): Scene to cast rays on.
            **kwargs: Additional keyword arguments are passed to
                cast_once.

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
                ray_dir (np.ndarray): Unit vectors for the rays. (Only
                    included by BouncedRayBlaster).
                ray_intensity (np.ndarray): Intensity of each ray. (Only
                    included by BouncedRayBlaster).
                original_ray_index (np.ndarray): Index of the original
                    ray that spawned each of the rays in this blaster.
                    (Only included by BouncedRayBlaster).
                bounces (dict): Raytrace results for bounces if
                    multibounce keyword argument is True.

        """
        return self.cast_once(
            scene, verbose_output=True, query_type=QueryType.INTERSECT,
            **kwargs
        )

    def compute_flux_density(self, scene, light_sources, **kwargs):
        r"""Compute the flux density on each scene element touched by
        this blaster from a set of light sources.

        Args:
            scene (hothouse.scene.Scene): Scene to get flux density for.
            light_sources (list): Set of RayBlasters used to determine
                the light incident on scene elements.
            **kwargs: Additional keyword arguments are passed to
                scene.compute_flux_density.

        Returns:
            array: Total flux density on surfaces intercepted by the
                rays.

        """
        fd_scene = scene.compute_flux_density(light_sources, **kwargs)
        out = np.zeros(self.nray, "f8")
        camera_hits = self.compute_count(scene)
        for ci, component in enumerate(scene.components):
            idx_ci = np.where(camera_hits["geomID"] == ci)[0]
            hits = camera_hits["primID"][idx_ci]
            out[idx_ci[hits >= 0]] += fd_scene[ci][hits[hits >= 0]]
        return out


class BouncedRayBlaster(RayBlaster):
    r"""Blaster created by bouncing rays off of a scene.

    Args:
        hits (dict): Raytracer results for the provided rays.
        prev_origins (np.ndarray): Origins of rays before the bounce.
        prev_directions (np.ndarray): Unit vectors describing ray
            directions before the bounce.
        prev_ray_intensity (np.ndarray, optional): Ray intensity before
            the bounce.
        prev_original_ray_index (np.ndarray, optional): Index of the
            original ray in the primary blaster that produced each of
            the rays prior to the bounce.
        ray_intensity_threshold_rel (float, optional): Relative threshold
            below which rays should no longer be tracked.
        ray_intensity_threshold_abs (float, optional): Absolute threshold
             below which rays should no longer be tracked.

    """

    # TODO: Multispectral bounces

    hits = traitlets.Dict()
    prev_origins = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f8"))
    prev_directions = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f8"))
    prev_ray_intensity = traittypes.Array().valid(check_dtype("f8"))
    prev_original_ray_index = traittypes.Array().valid(check_dtype("i4"))
    ray_intensity_threshold_rel = traitlets.CFloat()
    ray_intensity_threshold_abs = traitlets.CFloat()

    # @dependent_default("prev_ray_intensity", ["prev_nray"])
    # def _default_prev_ray_intensity(self):
    #     return np.ones((self.prev_nray, ), "f8")

    @dependent_default("prev_original_ray_index", ["prev_nray"])
    def _default_prev_original_ray_index(self):
        return np.arange(self.prev_nray, dtype="i4")

    @dependent_default(
        "ray_intensity_threshold_rel",
        ["prev_ray_intensity", "ray_intensity_threshold_abs"],
        strict=["ray_intensity_threshold_abs"],
    )
    def _default_ray_intensity_threshold_rel(self):
        if self.trait_has_value('ray_intensity_threshold_abs'):
            assert (self.prev_ray_intensity
                    == self.prev_ray_intensity[0]).all()
            return (self.ray_intensity_threshold_abs
                    / self.prev_ray_intensity[0])
        return 0.001

    @dependent_default(
        "ray_intensity_threshold_abs",
        ["prev_ray_intensity", "ray_intensity_threshold_rel"],
        strict=["ray_intensity_threshold_rel"],
    )
    def _default_ray_intensity_threshold_abs(self):
        assert (self.prev_ray_intensity
                == self.prev_ray_intensity[0]).all()
        return (self.ray_intensity_threshold_rel
                * self.prev_ray_intensity[0])

    @dependent_property("prev_origins")
    def prev_nray(self):
        r"""int: Number of rays in the previous blaster."""
        return self.prev_origins.shape[0]

    @dependent_property("hits", "prev_ray_intensity",
                        "ray_intensity_threshold_abs")
    def idx_hits(self):
        r"""np.ndarray: Boolean mask for rays that hit."""
        return np.logical_and(
            self.hits['primID'] >= 0,
            self.prev_ray_intensity >= self.ray_intensity_threshold_abs
        )

    @dependent_property("reflected_origins", "transmitted_origins",
                        "directions")
    def origins(self):
        r"""np.ndarray: Origins of the bounced rays."""
        out = np.vstack([
            self.reflected_origins,
            self.transmitted_origins,
        ])
        # Prevent intersection with origin by adding small offset
        out[:] += 1e-4 * self.directions
        return out

    @dependent_property("reflected_directions", "transmitted_directions")
    def directions(self):
        r"""np.ndarray: Directions of the bounced rays."""
        return np.vstack([
            self.reflected_directions,
            self.transmitted_directions,
        ])

    @dependent_property("reflected_intensity", "transmitted_intensity",
                        "idx_reflected", "idx_transmitted")
    def ray_intensity(self):
        r"""np.ndarray: Intensity of the bounced rays."""
        return np.hstack([
            self.reflected_intensity[self.idx_reflected],
            self.transmitted_intensity[self.idx_transmitted],
        ])

    @dependent_property("prev_original_ray_index", "idx_reflected",
                        "idx_transmitted")
    def original_ray_index(self):
        r"""np.ndarray: Index of the original ray in the original blaster
        producing each ray in this blaster."""
        return np.hstack([
            self.prev_original_ray_index[self.idx_reflected],
            self.prev_original_ray_index[self.idx_transmitted],
        ], dtype="i4")

    @dependent_property("prev_origins", "prev_directions",
                        "hits", "idx_hits")
    def intersections(self):
        r"""np.ndarray: Coordinates of ray intersections."""
        return (
            self.prev_origins[self.idx_hits]
            + sun_calc.op_along_axis(np.multiply,
                                     self.prev_directions[self.idx_hits],
                                     self.hits['tfar'][self.idx_hits])
        )

    @dependent_property("intersections", "idx_reflected", "idx_hits")
    def reflected_origins(self):
        r"""np.ndarray: Coordinates of reflected ray origins."""
        return self.intersections[self.idx_reflected[self.idx_hits]]

    @dependent_property("intersections", "idx_transmitted", "idx_hits")
    def transmitted_origins(self):
        r"""np.ndarray: Coordinates of transmitted ray origins."""
        return self.intersections[self.idx_transmitted[self.idx_hits]]

    @dependent_property("prev_directions", "idx_reflected", "hits")
    def reflected_directions(self):
        r"""np.ndarray: Unit vectors for the ray reflections."""
        # Rotate inverse of original direction 180 deg around surface
        # normal to get new direction
        return sun_calc.rotate_u(
            -self.prev_directions[self.idx_reflected], np.pi,
            self.hits['Ng'][self.idx_reflected]).astype("f8")

    @dependent_property("prev_directions", "idx_transmitted")
    def transmitted_directions(self):
        r"""np.ndarray: Unit vectors for the ray tranmissions."""
        # TODO: Refraction
        return self.prev_directions[self.idx_transmitted]

    @dependent_property("reflected_intensity",
                        "ray_intensity_threshold_abs")
    def idx_reflected(self):
        r"""np.ndarray: Boolean mask of previous rays that are
        reflected."""
        return (self.reflected_intensity
                >= self.ray_intensity_threshold_abs)

    @dependent_property("transmitted_intensity",
                        "ray_intensity_threshold_abs")
    def idx_transmitted(self):
        r"""np.ndarray: Boolean mask of previous rays that are
        transmitted."""
        return (self.transmitted_intensity
                >= self.ray_intensity_threshold_abs)

    @dependent_property("prev_nray", "hits", "idx_hits",
                        "prev_ray_intensity")
    def reflected_intensity(self):
        r"""np.ndarray: Intensity of reflected rays."""
        R = np.zeros((self.prev_nray, ), 'f8')
        if 'reflectance' in self.hits:
            R[self.idx_hits] = self.hits['reflectance'][self.idx_hits]
            # for i, reflectance in enumerate(self.reflectance):
            #     idx = (self.hits['geomID'] == i)
            #     R[idx] = reflectance[self.hits['primID'][idx]]
        return R * self.prev_ray_intensity

    @dependent_property("prev_nray", "hits", "idx_hits",
                        "prev_ray_intensity")
    def transmitted_intensity(self):
        r"""np.ndarray: Intensity of transmitted rays."""
        T = np.zeros((self.prev_nray, ), 'f8')
        if 'transmittance' in self.hits:
            T[self.idx_hits] = self.hits['transmittance'][self.idx_hits]
            # for i, transmittance in enumerate(self.transmittance):
            #     idx = (self.hits['geomID'] == i)
            #     T[idx] = transmittance[self.hits['primID'][idx]]
        return T * self.prev_ray_intensity

    def bounce(self, hits, **kwargs):
        r"""Create a ray tracer for bounced rays.

        Args:
            hits (dict, optional): Results from running this ray tracer.
            **kwargs: Additional keyword arguments are passed to
                BouncedRayBlaster.

        Returns:
            BouncedRayBlaster: Blaster containing ray information for the
                bounced rays.

        """
        kwargs.setdefault('prev_original_ray_index',
                          self.original_ray_index)
        for k in ['origins', 'directions', 'ray_intensity',
                  'original_ray_index']:
            kwargs.setdefault(f'prev_{k}', getattr(self, k))
        if 'ray_intensity_threshold_rel' not in kwargs:
            kwargs.setdefault('ray_intensity_threshold_abs',
                              self.ray_intensity_threshold_abs)
        return super(BouncedRayBlaster, self).bounce(hits, **kwargs)

    def cast_once(self, scene, verbose_output=True,
                  query_type=QueryType.INTERSECT, dists=None,
                  multibounce=False,
                  include_attributes=["reflectance", "transmittance"],
                  **kwargs):
        r"""Run the embree raytrace once on a scene for this bounce.
        Note that many of the defaults for this method differ than those
        for the base RayBlaster due to the way it is intended to be used.

        Args:
            scene (hothouse.scene.Scene): Scene to raytrace.
            verbose_output (bool, optional): If True, output a dictionary
                of all ray properties.
            query_type (QueryType, optional): Raytrace query type
                controlling the output.
            dists (np.ndarray, optional): Array of distances for each
                ray that should be updated.
            multibounce (bool, optional): If True, rays should be tracked
                through reflections/transmission.
            include_attributes (bool, list): If True, include all face
                attributes for each intercepted face in the results. If
                a list is provided, just the specified attributes will
                be included. This will be ignored if verbose_output is
                False.
            **kwargs: Additional keyword arguments are passed to
                BouncedRayBlaster if multibounce is True.

        Returns:
            np.ndarray, dict: Ray trace result.

        """
        out = super(BouncedRayBlaster, self).cast_once(
            scene, verbose_output=verbose_output,
            query_type=query_type, dists=dists, multibounce=multibounce,
            include_attributes=include_attributes, **kwargs
        )
        if isinstance(out, dict):
            out.update(
                ray_dir=self.directions,
                ray_intensity=self.ray_intensity,
                original_ray_index=self.original_ray_index,
            )
        return out

    @classmethod
    def resize_bounces(cls, bounces, maxbounce):
        r"""Resize bounce output data to match a new number of maximum
        bounces.

        Args:
            bounces (dict): Joined output from raytrace for a set of
                bounces.
            maxbounce (int): New number of maximum bounces.

        """
        prev_maxbounce = max(bounces['nbounce'])
        if maxbounce == prev_maxbounce:
            return
        assert maxbounce > prev_maxbounce
        for k in list(bounces.keys()):
            if k == 'nbounce':
                continue
            prev_shape = bounces[k].shape
            shape = list(prev_shape)
            shape[1] = maxbounce
            bounces[k] = np.resize(bounces[k], shape)
            if maxbounce > prev_maxbounce:
                bounces[k][:, prev_shape[1]:, ...] = (
                    cls._null_field_values.get(k, 0)
                )

    def join_bounces(self, bounces):
        r"""Combine output from a set of bounces.

        Args:
            bounces (list): Output from raytrace for each bounce.

        Returns:
            dict: Combined data from the bounces.

        """
        maxbounce = len(bounces)
        nray = self.prev_nray
        nbounce = np.zeros(nray, dtype="i4")
        for x in bounces:
            idx, idx_count = np.unique(x['original_ray_index'],
                                       return_counts=True)
            nbounce[idx] += idx_count
        maxbounce = max(nbounce)
        out = {
            'nbounce': np.zeros(nray, dtype="i4"),
            'primID': np.empty((nray, maxbounce), dtype="i4"),
            'geomID': np.empty((nray, maxbounce), dtype="i4"),
            'tfar': np.empty((nray, maxbounce), dtype="f4"),
            'u': np.empty((nray, maxbounce), dtype="f4"),
            'v': np.empty((nray, maxbounce), dtype="f4"),
            'Ng': np.empty((nray, maxbounce, 3), dtype="f4"),
            'ray_dir': np.empty((nray, maxbounce, 3), dtype="f8"),
            'ray_intensity': np.empty((nray, maxbounce), dtype="f8"),
        }
        for k in self.hits.keys():
            if k not in out:
                out[k] = np.empty(
                    (nray, maxbounce, *self.hits[k].shape[1:]),
                    dtype="f8",
                )
            out[k].fill(self._null_field_values.get(k, 0))
        for x in bounces:
            idx_ray = x['original_ray_index']
            idx_bounce = np.zeros(idx_ray.shape, dtype="i4")
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

    center = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    forward = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    up = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    east = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    width = traitlets.CFloat(1.0)
    height = traitlets.CFloat(1.0)
    nx = traitlets.CInt(512)
    ny = traitlets.CInt(512)
    period = traittypes.Array(
        np.zeros((3,), "i4")
    ).valid(check_dtype("i4"), check_shape(3))
    intensity = traitlets.CFloat()
    intensity_density = traitlets.CFloat()

    @dependent_default("intensity", [
        "ray_intensity", "intensity_density", "width", "height",
    ])
    def _default_intensity(self):
        if self.trait_has_value("ray_intensity"):
            return self.ray_intensity.sum()
        if not self.trait_has_value("intensity_density"):
            return 1.0
        return self.intensity_density * self.width * self.height

    @dependent_default("intensity_density", [
        "intensity", "width", "height",
    ])
    def _default_intensity_density(self):
        return self.intensity / (self.width * self.height)

    @dependent_default("forward", ["up", "east"])
    def _default_forward(self):
        if not (self.trait_has_value("up")
                and self.trait_has_value("east")):
            raise traitlets.TraitError(
                "At least two direction must be provided "
                "out of forward, up, & east")
        return np.cross(self.up, self.east)

    @dependent_default("up", ["east", "forward"])
    def _default_up(self):
        return np.cross(self.east, self.forward)

    @dependent_default("east", ["forward", "up"])
    def _default_east(self):
        return np.cross(self.forward, self.up)

    @dependent_default("directions", ["nx", "ny", "forward"])
    def _default_directions(self):
        self._directions = np.zeros((self.nx, self.ny, 3), dtype="f8")
        self._directions[:] = self.forward[None, None, :]
        return self._directions.view().reshape((self.nx * self.ny, 3))

    @dependent_default("origins", [
        "nx", "ny", "width", "height", "center", "east", "up"
    ])
    def _default_origins(self):
        # here origin is not the center, but the bottom left
        self._origins = np.zeros((self.nx, self.ny, 3), dtype="f8")
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

    @dependent_property("period")
    def is_periodic(self):
        r"""bool: True if the blaster will be replicated periodically."""
        return np.any(self.period > 0)

    def cast_once(self, scene, verbose_output=False,
                  query_type=QueryType.DISTANCE, dists=None, **kwargs):
        r"""Run the embree raytrace once on a scene.

        Args:
            scene (hothouse.scene.Scene): Scene to raytrace.
            verbose_output (bool, optional): If True, output a dictionary
                of all ray properties.
            query_type (QueryType, optional): Raytrace query type
                controlling the output.
            dists (np.ndarray, optional): Array of distances for each
                ray that should be updated.
            **kwargs: Additional keyword arguments are passed to the
                RayBlaster method.

        Returns:
            np.ndarray, dict: Ray trace result.

        """
        prev_verbose_output = verbose_output
        if self.is_periodic:
            if dists is None:
                dists = np.empty(
                    kwargs.get('origins', self.origins).shape[0],
                    'f4')
                dists.fill(self._null_field_values["tfar"])
            if query_type != QueryType.OCCLUDED:
                verbose_output = True
        out = super(OrthographicRayBlaster, self).cast_once(
            scene, verbose_output=verbose_output,
            query_type=query_type, dists=dists, **kwargs
        )
        if not self.is_periodic:
            return out
        from hothouse.scene import PeriodicScene
        shifts = PeriodicScene.get_periodic_shifts(
            np.array([self.width + self.width / self.nx,
                      self.height + self.height / self.ny, 0.0]),
            np.vstack([self.east, self.up, self.forward]),
            self.period)
        tfar0 = None
        if isinstance(out, dict):
            tfar0 = out['tfar'].copy()
        maxbounce = 0
        if 'bounces' in out:
            maxbounce = max(out['bounces']['nbounce'])
        for shift in shifts:
            iblaster = RayBlaster(
                origins=(self.origins + shift),
                directions=self.directions,
                ray_intensity=self.ray_intensity,
            )
            iout = iblaster.cast_once(
                scene, verbose_output=verbose_output,
                query_type=query_type, dists=dists, **kwargs
            )
            if query_type == QueryType.OCCLUDED:
                idx = (iout == 0)
                out[idx] = 0
            else:
                idx = (iout['tfar'] < tfar0)
                if not idx.any():
                    continue
                tfar0[idx] = iout['tfar'][idx]
                if 'bounces' in out:
                    adjust_out = iout
                    if max(iout['bounces']['nbounce']) > maxbounce:
                        maxbounce = max(iout['bounces']['nbounce'])
                        adjust_out = out
                    BouncedRayBlaster.resize_bounces(
                        adjust_out['bounces'], maxbounce)
                for k, v in iout.items():
                    if k == 'bounces':
                        for kb, vb in v.items():
                            out[k][kb][idx, ...] = vb[idx, ...]
                    else:
                        out[k][idx, ...] = v[idx, ...]
        if query_type != QueryType.OCCLUDED and not prev_verbose_output:
            if query_type == QueryType.DISTANCE:
                out = out['tfar']
            elif query_type == QueryType.INTERSECT:
                out = out['primID']  # This is what pyembree returns
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

    ground = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    zenith = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    north = traittypes.Array().valid(check_dtype("f8"), check_shape(3))

    solar_altitude = traitlets.CFloat()
    solar_azimuth = traitlets.CFloat()
    solar_distance = traitlets.CFloat()
    scene_limits = traittypes.Array(None, allow_none=True).valid(
        check_shape(None, 3), check_dtype("f8"))

    altitude = traitlets.Float()
    pressure = traitlets.Float()
    temperature = traitlets.Float(12.0)
    eta_par = traitlets.Float(0.368)
    eta_photon = traitlets.Float(4.56)

    @dependent_default("forward", [
        "zenith_direction", "ground", "north",
        "solar_altitude", "solar_azimuth",
    ])
    def _default_forward(self):
        # Negative to point from sun to the earth rather than from
        # earth to the sun
        return -self.solar_rotation(self.zenith_direction, is_ray=True)

    @dependent_default("up", [
        "solar_east",
        "zenith_direction", "ground", "north",
        "solar_altitude", "solar_azimuth",
    ])
    def _default_up(self):
        # The "east" used here is not the "east" used elsewhere.
        # This is the east wrt north etc, but we need an east
        # for blasting from elsewhere.
        return -self.solar_rotation(self.solar_east, is_ray=True)

    @dependent_property(
        "date", "latitude", "longitude", "pressure", "temperature",
    )
    def solpos_info(self):
        r"""pandas.DataFrame: Information about the sun's position at
        the location for the provided date/time and location information.
        """
        return pvlib.solarposition.get_solarposition(
            self.date, self.latitude, self.longitude,
            pressure=self.pressure, temperature=self.temperature,
        )

    @dependent_default("solar_altitude", ["solpos_info"])
    def _default_solar_altitude(self):
        solar_altitude = self.solpos_info["apparent_elevation"].iloc[0]
        if solar_altitude < 0:
            raise traitlets.TraitError(
                f"For the provided lat ({self.latitude}), "
                f"long ({self.longitude}), date, & time "
                f"({self.date}) the sun will be below the horizon."
            )
        return solar_altitude

    @traitlets.validate("solar_altitude")
    def _validate_solar_altitude(self, proposal):
        if proposal['value'] < 0:
            raise traitlets.TraitError(
                "The solar altitude must be positive for the sun "
                "to be above the horizon"
            )
        return proposal['value']

    @dependent_default("solar_azimuth", ["solpos_info"])
    def _default_solar_azimuth(self):
        return self.solpos_info["azimuth"].iloc[0]

    @dependent_default("solar_distance", ["zenith", "ground"])
    def _default_solar_distance(self):
        zd_nonorm = self.zenith - self.ground
        return np.linalg.norm(zd_nonorm)

    @dependent_default("width", ["scene_limits", "limits_east"])
    def _default_width(self):
        out = 1.0
        if self.scene_limits is not None:
            out = self.limits_east[1] - self.limits_east[0]
        return out

    @dependent_default("height", ["scene_limits", "limits_up"])
    def _default_height(self):
        out = 1.0
        if self.scene_limits is not None:
            out = self.limits_up[1] - self.limits_up[0]
        return out

    @dependent_default("center", [
        "scene_limits", "limits_east", "limits_up", "east", "up",
        "solar_distance", "forward", "ground", "height",
        "solar_altitude",
    ])
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
                    * sun_calc.stable_tan(np.radians(self.solar_altitude))
                )
            ),
        )
        v = v + offset * self.up
        return v

    @dependent_default("altitude", ["pressure"])
    def _default_altitude(self):
        if not self.trait_has_value("pressure"):
            return 0.0
        import pvlib
        return pvlib.atmosphere.pres2alt(self.pressure)

    @dependent_default("pressure", ["altitude"])
    def _default_pressure(self):
        import pvlib
        return pvlib.atmosphere.alt2pres(self.altitude)

    @dependent_default("intensity", [
        "ray_intensity", "intensity_density", "width", "height",
    ])
    def _default_intensity(self):
        if not (self.trait_has_value("ray_intensity")
                or self.trait_has_value("intensity_density")):
            # Force default
            self.intensity_density
        return super(SunRayBlaster, self)._default_intensity()

    @dependent_default("intensity_density", [
        "intensity", "width", "height", "solar_ppfd",
    ])
    def _default_intensity_density(self):
        if not (self.trait_has_value("intensity")
                or self.trait_has_value("ray_intensity")):
            return self.solar_ppfd['direct']
        return super(SunRayBlaster, self)._default_intensity_density()

    @dependent_default("diffuse_intensity", ["solar_ppfd"])
    def _default_diffuse_intensity(self):
        return self.solar_ppfd['diffuse']

    @dependent_property(
        "latitude", "longitude", "date", "altitude", "pressure",
        "temperature", "eta_par", "eta_photon",
    )
    def solar_ppfd(self):
        r"""dict: Solar intensity calculated based on the location."""
        return sun_calc.solar_ppfd(
            self.latitude, self.longitude, self.date,
            altitude=self.altitude, pressure=self.pressure,
            temperature=self.temperature, eta_par=self.eta_par,
            eta_photon=self.eta_photon,
        )

    @classmethod
    def get_solar_direction(cls, latitude, longitude, date, up, north,
                            **kwargs):
        r"""Get the direction that rays from the sun will travel for
        a given location, time, and orientation.

        Args:
            latitude (float): Location latitude (in degrees).
            longitude (float): Location longitude (in degrees).
            date (datetime.datetime): Date/time.
            up (np.ndarray, optional): Normal unit vector for the ground.
            north (np.ndarray): Unit vector for the north cardinal
                direction.
            **kwargs: Additional keyword arguments are passed to
                SunRayBlaster instance.

        Returns:
            np.ndarray: Unit vector in the direction of sun rays.

        """
        instance = cls(
            latitude=latitude, longitude=longitude,
            date=date, zenith=(10 * up / np.linalg.norm(up)),
            ground=np.zeros((3,), "f8"),
            north=north,
            **kwargs
        )
        return instance.forward

    @dependent_property("zenith", "ground")
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
        # TODO: Move rotation matrices into dependent_property
        return sun_calc.rotate_u(
            sun_calc.rotate_u(point - origin,
                              np.radians(90 - self.solar_altitude),
                              self.north),
            np.radians(90 - self.solar_azimuth),
            self.zenith_direction,
        ).astype("f8") + origin

    @dependent_property("scene_limits", "east")
    def limits_east(self):
        r"""Limits projected in the east direction."""
        adotb = (np.dot(self.scene_limits, self.east)
                 / np.linalg.norm(self.east))
        return [adotb.min(), adotb.max()]

    @dependent_property("scene_limits", "up")
    def limits_up(self):
        r"""Limits projected in the up direction."""
        adotb = (np.dot(self.scene_limits, self.up)
                 / np.linalg.norm(self.up))
        return [adotb.min(), adotb.max()]

    @dependent_property("north", "zenith_direction")
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

    """
    fov_width = traitlets.CFloat(90.0)
    fov_height = traitlets.CFloat(90.0)
    center = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    forward = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    up = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    east = traittypes.Array().valid(check_dtype("f8"), check_shape(3))
    width = traitlets.CFloat(1.0)
    height = traitlets.CFloat(1.0)
    nx = traitlets.CInt(512)
    ny = traitlets.CInt(512)

    @dependent_default("east", ["forward", "up"])
    def _default_east(self):
        return np.cross(self.forward, self.up)

    @dependent_default("directions", ["origins", "camera_origin"])
    def _default_directions(self):
        self._directions = self.origins - self.camera_origin
        self._directions = sun_calc.norm_along_axis(
            self._directions, 1).astype("f8")
        return self._directions.view()

    @dependent_default("origins", [
        "nx", "ny", "width", "height", "center", "east", "up",
    ])
    def _default_origins(self):
        # here origin is not the center, but the bottom left
        self._origins = np.zeros((self.nx, self.ny, 3), dtype="f8")
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

    @dependent_property("width", "fov_width")
    def camera_distance(self):
        r"""float: Distance of the camera from the image plane."""
        return (
            (self.width / 2.0)
            / sun_calc.stable_tan(np.radians(self.fov_width / 2.0))
        )

    @dependent_property("center", "camera_distance", "forward")
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

    @dependent_default("directions", [
        "nx", "ny", "forward", "east", "fov_width", "fov_height",
    ])
    def _default_directions(self):
        self._directions = np.zeros((self.nx * self.ny, 3), dtype="f8")
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
        self._directions = self._directions.astype("f8")
        return self._directions.view()

    @dependent_default("origins", ["nx", "ny", "center"])
    def _default_origins(self):
        # here origin is not the center, but the bottom left
        if self.dont_include_center:
            self._origins = np.zeros((self.nx, self.ny, 3), dtype="f8")
            self._origins[:] = self.center[None, None, :]
            return self._origins.view().reshape((self.nx * self.ny, 3))
        self._origins = np.zeros((self.nx * self.ny + 1, 3), dtype="f8")
        self._origins[:] = self.center[None, None, :]
        return self._origins.view()
