from enum import Enum
import numpy as np

import pyembree
import traitlets
import traittypes
import datetime
import pvlib

from .traits_support import check_dtype, check_shape

from hothouse import sun_calc
from .ray_callbacks import RayCollisionPrinter, RayCollisionMultiBounce

# pyembree receives origins and directions.


class QueryType(Enum):
    DISTANCE = "DISTANCE"
    OCCLUDED = "OCCLUDED"
    INTERSECT = "INTERSECT"


class RayBlaster(traitlets.HasTraits):
    origins = traittypes.Array().valid(check_shape(None, 3), check_dtype("f4"))
    directions = traittypes.Array().valid(check_shape(None, 3), check_dtype("f4"))
    intensity = traitlets.CFloat(1.0)
    diffuse_intensity = traitlets.CFloat(0.0)
    multibounce = traitlets.CBool(False)

    @property
    def ray_intensity(self):
        r"""float: Intensity of single ray."""
        return self.intensity / self.origins.shape[0]

    def cast_once(self, scene, verbose_output=False, query_type=QueryType.DISTANCE):
        if self.multibounce:
            callback_handler = RayCollisionMultiBounce(self.origins.shape[0], 10)
        else:
            callback_handler = None
        output = scene.embree_scene.run(
            self.origins,
            self.directions,
            query=query_type._value_,
            output=verbose_output,
            callback_handler=callback_handler
        )
        if self.multibounce:
            output['bounces'] = callback_handler.bounces
        return output

    def compute_distance(self, scene):
        output = self.cast_once(
            scene, verbose_output=False, query_type=QueryType.DISTANCE
        )
        return output

    def compute_count(self, scene):
        output = self.cast_once(
            scene, verbose_output=True, query_type=QueryType.INTERSECT
        )
        return output

    def compute_flux_density(self, scene, light_sources, any_direction=True):
        r"""Compute the flux density on each scene element touched by
        this blaster from a set of light sources.

        Args:
            scene (Scene): Scene to get flux density for.
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
    center = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    forward = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    up = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    east = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    width = traitlets.CFloat(1.0)
    height = traitlets.CFloat(1.0)
    nx = traitlets.CInt(512)
    ny = traitlets.CInt(512)

    @traitlets.default("east")
    def _default_east(self):
        return np.cross(self.forward, self.up)

    def __init__(self, *args, **kwargs):
        super(OrthographicRayBlaster, self).__init__(*args, **kwargs)

        # here origin is not the center, but the bottom left
        self._directions = np.zeros((self.nx, self.ny, 3), dtype="f4")
        self._directions[:] = self.forward[None, None, :]
        self.directions = self._directions.view().reshape((self.nx * self.ny, 3))

        self._origins = np.zeros((self.nx, self.ny, 3), dtype="f4")
        offset_x, offset_y = np.mgrid[
            -self.width / 2 : self.width / 2 : self.nx * 1j,
            -self.height / 2 : self.height / 2 : self.ny * 1j,
        ]
        self._origins[:] = (
            self.center
            + offset_x[..., None] * self.east
            + offset_y[..., None] * self.up
        )
        self.origins = self._origins.view().reshape((self.nx * self.ny, 3))


class SunRayBlaster(OrthographicRayBlaster):
    # ground: Position of center of ray projection on the ground
    # zenith: Position directly above 'ground' at distance that sun
    #     blaster should be placed.
    # north: Direction of north on ground from 'ground'

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
    multibounce = traitlets.CBool(True)

    @traitlets.default("_solpos_info")
    def _solpos_info_default(self):
        return pvlib.solarposition.get_solarposition(
            self.date, self.latitude, self.longitude
        )

    @traitlets.default("solar_altitude")
    def _default_solar_altitude(self):
        solar_altitude = self._solpos_info["apparent_elevation"][0]
        if solar_altitude < 0:
            raise ValueError(
                "For the provided lat, long, date, & time "
                "the sun will be below the horizon."
            )
        return solar_altitude

    @traitlets.default("solar_azimuth")
    def _default_solar_azimuth(self):
        return self._solpos_info["azimuth"][0]

    @property
    def zenith_direction(self):
        zd_nonorm = self.zenith - self.ground
        solar_distance = np.linalg.norm(zd_nonorm)
        print("sd zd", solar_distance, zd_nonorm)
        return zd_nonorm / solar_distance

    @traitlets.default("solar_distance")
    def _solar_distance_default(self):
        zd_nonorm = self.zenith - self.ground
        return np.linalg.norm(zd_nonorm)

    def solar_rotation(self, point):
        r"""Rotate a point according to same rotation that moves
        sun from the zenith to it location in the sky.

        Args:
            point (array): 3D point to rotate

        """
        return sun_calc.rotate_u(
            sun_calc.rotate_u(point, np.radians(90 - self.solar_altitude), self.north),
            np.radians(90 - self.solar_azimuth),
            self.zenith_direction,
        )

    @traitlets.default("forward")
    def _forward_default(self):
        # Negative to point from sun to the earth rather than from
        # eart to the sun
        return -self.solar_rotation(self.zenith_direction)

    @traitlets.default("center")
    def _center_default(self):
        v = self.ground - self.solar_distance * self.forward
        offset = max(
            0.0,
            (
                (self.height / 2.0)
                - np.abs(
                    np.linalg.norm(v - self.ground)
                    * np.tan(np.radians(self.solar_altitude))
                )
            )
            / 2,
        )
        print(offset)
        v = v + offset * self.up
        return v

    @traitlets.default("up")
    def _up_default(self):
        zenith_direction = self.zenith_direction
        east = np.cross(self.north, zenith_direction)
        # The "east" used here is not the "east" used elsewhere.
        # This is the east wrt north etc, but we need an east for blasting from elsewhere.
        return -self.solar_rotation(east)


class ProjectionRayBlaster(RayBlaster):
    pass


class SphericalRayBlaster(RayBlaster):
    pass
