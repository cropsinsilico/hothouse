import traittypes
import traitlets
import pythreejs
import numpy as np
import functools
import pvlib
# from IPython.core.display import display

from .model import Model
from .blaster import (
    QueryType, RayBlaster, OrthographicRayBlaster, SunRayBlaster,
)
from .traits_support import check_shape, check_dtype
from .sun_calc import solar_ppfd

from embreex.mesh_construction import TriangleMesh

# Use native embreex scene
from embreex import rtcore_scene as rtcs
EmbreeScene = rtcs.EmbreeScene

# Use local subclass of embreex scene w/ support for callbacks
# from .callback_handler import CallbackScene
# EmbreeScene = CallbackScene

cached_property = getattr(functools, "cached_property", property)


class Scene(traitlets.HasTraits):
    r"""Container for component geometries that will be traced
    and properties of the context in which they reside.

    Args:
        ground (np.ndarray, optional): Scene center.
        up (np.ndarray, optional): Normal unit vector for the ground.
        north (np.ndarray, optional): Unit vector for the north cardinal
            direction.
        components (list, optional): 3D components in the scene.
        meshes (list, optional): 3D geometries for the components in the
            scene.
        embree_scene (pyembree.EmbreeScene, optional):

    """

    ground = traittypes.Array(np.array([0.0, 0.0, 0.0], "f4")).valid(
        check_dtype("f4"), check_shape(3))
    up = traittypes.Array(np.array([0.0, 0.0, 1.0], "f4")).valid(
        check_dtype("f4"), check_shape(3))
    north = traittypes.Array(np.array([0.0, 1.0, 0.0], "f4")).valid(
        check_dtype("f4"), check_shape(3))
    components = traitlets.List(trait=traitlets.Instance(Model))
    meshes = traitlets.List(trait=traitlets.Instance(TriangleMesh))
    embree_scene = traitlets.Instance(EmbreeScene, args=tuple())

    # TODO: Add surface for ground so that reflection from ground
    # is taken into account

    def post_cast(self, query_type, output):
        r"""Finalize the results from running the ray tracer.

        Args:
            query_type (QueryType): Raytrace query type of the output.
            output (object): Raytracer result.

        Returns:
            object: Finalized raytracer result.

        """
        return output

    def add_component(self, component):
        r"""Add a component to the scene.

        Args:
            component (hothouse.model.Model): 3D component.

        """
        # Force traitlet update
        self.components = self.components + [component]
        self.meshes.append(
            TriangleMesh(self.embree_scene, component.triangles))

    def compute_hit_count(self, blaster):
        r"""Run the raytracer to determine how many rays will hit each
        component face in the scene.

        Args:
            blaster (hothouse.blaster.RayBlaster): Blaster containing
                rays to trace.

        Returns:
            dict: Mapping between component index and arrays of hit
                counts for each face in the component geometry.

        """
        output = blaster.compute_count(self)
        component_counts = {}
        for ci, component in enumerate(self.components):
            hits = output["primID"][output["geomID"] == ci]
            component_counts[ci] = np.bincount(
                hits[hits >= 0], minlength=component.triangles.shape[0]
            )
        return component_counts

    @property
    def ncomponents(self):
        r"""int: Number of components in the scene."""
        return len(self.components)

    @property
    def transmittance(self):
        r"""list: Tranmittance values for each component's faces."""
        return [c.transmittance for c in self.components]

    @property
    def reflectance(self):
        r"""list: Reflectance values for each component's faces."""
        return [c.reflectance for c in self.components]

    @cached_property
    def limits(self):
        r"""np.ndarray: Positions of corners of a box containing all
        points in the scene."""
        mins = []
        maxs = []
        for c in self.components:
            mins.append(np.min(c.vertices, axis=0))
            maxs.append(np.max(c.vertices, axis=0))
        mins = np.min(np.vstack(mins), axis=0)
        maxs = np.max(np.vstack(maxs), axis=0)
        limits = np.vstack([mins, maxs])
        xx, yy, zz = np.meshgrid(limits[:, 0], limits[:, 1], limits[:, 2])
        limits = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
        return limits

    def compute_solar_ppfd(self, latitude, longitude, date,
                           direct_ppfd=None, diffuse_ppfd=None,
                           any_direction=True, **kwargs):
        r"""Compute the photon flux density on each face in the scene
        from solar irradiance for a given location and time/date.

        Args:
            latitude (float): Latitude (in degrees) of the scene.
            longitude (float): Longitude (in degrees) of the scene.
            date (datetime.datetime): Time when PPFD should be calculated.
                This determines the incidence angle of light from the
                sun.
            direct_ppfd (float, optional): Direct Photosynthetic
                Photon Flux Density (PPFD) at the surface of the
                Earth for the specified location and time. If not
                provided, the direct_ppfd will be calculated based on the
                location and time/date.
            diffuse_ppfd (float, optional): Diffuse Photosynthetic
                Photon Flux Density (PPFD) at the surface of the
                Earth for the specified location and time. If not
                provided, the diffuse_ppfd will be calculated based on
                the location and time/date.
            any_direction (bool, optional): If True, light is deposited
                on component reguardless of if the blaster rays hit the
                front or back of a component surface. If False, light
                is only deposited if the blaster rays hit the front.
                Defaults to True.
            **kwargs: Additional keyword arguments are passed to the
                SunRayBlaster constructor.

        Returns:
            dict: Mapping from scene component to an array of photon flux
                density values for each triangle in the component.

        """
        rb = self.get_sun_blaster(latitude, longitude, date,
                                  direct_ppfd=direct_ppfd,
                                  diffuse_ppfd=diffuse_ppfd, **kwargs)
        return self.compute_flux_density(rb, any_direction=any_direction)

    def get_sun_blaster(self, latitude, longitude, date,
                        direct_ppfd=None, diffuse_ppfd=None,
                        zenith=None, **kwargs):
        r"""Get a sun blaster that is adjusted for this scene so that
        the blaster will never intercept a component in the scene. This
        distance is determined by computing the maximum distance of any
        vertex in the scene from the ground parameter.

        Args:
            latitude (float): Latitude (in degrees) of the scene.
            longitude (float): Longitude (in degrees) of the scene.
            date (datetime.datetime): Time when PPFD should be calculated.
                This determines the incidence angle of light from the
                sun.
            direct_ppfd (float, optional): Direct Photosynthetic
                Photon Flux Density (PPFD) at the surface of the
                Earth for the specified location and time. If not
                provided, the direct_ppfd will be calculated based on the
                location and time/date.
            diffuse_ppfd (float, optional): Diffuse Photosynthetic
                Photon Flux Density (PPFD) at the surface of the
                Earth for the specified location and time. If not
                provided, the diffuse_ppfd will be calculated based on
                the location and time/date.
            zenith (np.ndarray): Position directly above 'ground' at
                distance that sun blaster should be placed.
            **kwargs: Additional keyword arguments are passed to the
                SunRayBlaster constructor.

        Returns:
            SunRayBlaster: Blaster tuned to this scene.

        """
        if zenith is None:
            max_distance2 = 0.0
            for c in self.components:
                max_distance2 = max(
                    max_distance2,
                    np.max(np.sum((c.vertices-self.ground)**2, axis=1)))
            max_distance = np.sqrt(max_distance2)
            zenith = self.up * max_distance + self.ground
        if direct_ppfd is not None:
            kwargs['intensity_density'] = direct_ppfd
        if diffuse_ppfd is not None:
            kwargs['diffuse_intensity'] = diffuse_ppfd
        kwargs.setdefault('scene_limits', self.limits)
        blaster = SunRayBlaster(latitude=latitude,
                                longitude=longitude, date=date,
                                ground=self.ground, north=self.north,
                                zenith=zenith, **kwargs)
        return blaster

    def animate_sun(self, camera, latitude, longitude,
                    t_start, t_stop, n_step, altitude=180.0,
                    fname=None):
        r"""Create an animation of the sun moving across the scene
        during the specified time.

        Args:
            latitude (float): Latitude in degrees.
            longitude (float): Longitude in degrees.
            t_start (datetime.datetime): Start date & time w/ timezone
                information.
            t_stop (datetime.datetime): Stop date & time w/ timezone
                information.
            n_step (int): Number of steps between t_start and t_stop
                to include in animation.
            altitude (float, optional): Distance above sea level in
                meters. Defaults to 180 meters (roughly the average for
                Illinois).
            fname (str, optional): File where animation should be saved.
                Defaults to None and animation will be shown instead.

        Returns:
            matplotlib.animation.FuncAnimation: Animation object.

        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, writers
        from matplotlib.colors import LogNorm
        import pandas as pd
        nx = ny = 1024
        fig, ax = plt.subplots()
        img = plt.imshow(np.nan * np.ones((nx, ny)), origin='lower',
                         norm=LogNorm(50, 5.0e4))
        plt.colorbar()

        def update(frame):
            ppfd_tot = solar_ppfd(latitude, longitude, frame,
                                  altitude=altitude)
            sun = self.get_sun_blaster(latitude, longitude, frame,
                                       nx=nx, ny=ny,
                                       direct_ppfd=ppfd_tot['direct'],
                                       diffuse_ppfd=ppfd_tot['diffuse'],
                                       multibounce=True)
            o = camera.compute_flux_density(self, sun)
            o[o <= 0] = np.nan
            img.set_data(o.reshape((camera.ny, camera.nx), order='F'))
            return img,

        dates = pd.date_range(t_start, t_stop, periods=n_step)
        ani = FuncAnimation(fig, update, frames=list(dates), blit=False)
        if fname is None:
            plt.show()
        else:
            Writer = writers['html']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(fname, writer=writer)
        return ani

    def compute_flux_density(self, light_sources, any_direction=True):
        r"""Compute the flux density on each scene element from a
        set of light sources. Values will be calculated from the
        'intensity' attribute of the light source blasters such that
        the flux density will have units of

            [intensity units] / [distance unit from scene] ** 2.

        Args:
            light_sources (list): Set of RayBlasters used to determine
                the light incident on scene elements.
            any_direction (bool, optional): If True, light is deposited
                on component reguardless of if the blaster rays hit the
                front or back of a component surface. If False, light
                is only deposited if the blaster rays hit the front.
                Defaults to True.

        Returns:
            dict: Mapping from scene component to an array of flux
                density values for each triangle in the component.

        """
        if isinstance(light_sources, RayBlaster):
            light_sources = [light_sources]
        component_fd = {}
        for ci, component in enumerate(self.components):
            component_fd[ci] = np.zeros(component.triangles.shape[0], "f4")
        for blaster in light_sources:
            counts = blaster.compute_count(self)
            if blaster.multibounce:
                orthographic = isinstance(blaster, OrthographicRayBlaster)
                for i in range(max(counts["bounces"]["nbounce"])):
                    orthographic = (orthographic and (i == 0))
                    if orthographic:
                        ray_dir = blaster.forward
                        ray_intensity = blaster.ray_intensity
                        diffuse_intensity = blaster.diffuse_intensity
                    else:
                        ray_dir = counts["bounces"]["ray_dir"][:, i, :]
                        ray_intensity = (
                            blaster.ray_intensity
                            * counts["bounces"]["power"][:, i])
                        diffuse_intensity = 0.0
                    primID = counts["bounces"]["primID"][:, i]
                    geomID = counts["bounces"]["geomID"][:, i]
                    self._accumulate_hits(component_fd, primID, geomID,
                                          ray_dir, ray_intensity,
                                          diffuse_intensity,
                                          orthographic=orthographic,
                                          any_direction=any_direction)
            else:
                if isinstance(blaster, OrthographicRayBlaster):
                    ray_dir = blaster.forward
                    orthographic = True
                else:
                    ray_dir = blaster.directions
                    orthographic = False
                self._accumulate_hits(component_fd, counts["primID"],
                                      counts["geomID"], ray_dir,
                                      blaster.ray_intensity,
                                      blaster.diffuse_intensity,
                                      orthographic=orthographic,
                                      any_direction=any_direction)
        return component_fd

    def _calc_incident_power(self, ray_dir, norm, area, any_direction=True):
        aoi = np.arccos(
            np.dot(norm, -ray_dir) / (2.0 * area * np.linalg.norm(ray_dir)))
        if isinstance(aoi, np.ndarray):
            if any_direction:
                aoi[aoi > np.pi/2] -= np.pi
            else:
                aoi[aoi > np.pi/2] = np.pi / 2  # No contribution
        else:
            if aoi > np.pi/2:
                if any_direction:
                    aoi -= np.pi
                else:
                    aoi = np.pi / 2
        out = np.cos(aoi) / area
        return out

    def _accumulate_hits(self, component_fd, primID, geomID,
                         ray_dir, ray_intensity, diffuse_intensity,
                         orthographic=False, any_direction=True):
        any_hits = (primID >= 0)
        for ci, component in enumerate(self.components):
            norms = component.normals
            areas = component.areas
            idx_hits = np.logical_and(geomID == ci, any_hits)
            if orthographic:
                component_counts = np.bincount(
                    primID[idx_hits], minlength=component.triangles.shape[0])
                component_fd[ci] += np.array(
                    component_counts * ray_intensity
                    * self._calc_incident_power(
                        ray_dir, norms, areas,
                        any_direction=any_direction))
            else:
                if not isinstance(ray_intensity, np.ndarray):
                    ray_intensity = ray_intensity * np.ones(primID.shape)
                # TODO: This loop can be removed if AOI is calculated
                # for each intersection by embree (or callback)
                for idx_ray in np.where(idx_hits)[0]:
                    idx_scene = primID[idx_ray]
                    component_fd[ci][idx_scene] += (
                        ray_intensity[idx_ray]
                        * self._calc_incident_power(
                            ray_dir[idx_ray, :],
                            norms[idx_scene], areas[idx_scene],
                            any_direction=any_direction))
            # Diffuse
            # TODO: This assumes diffuse light comes from everywhere
            if diffuse_intensity > 0.0:
                tilt = np.arccos(
                    np.dot(norms, self.up)
                    / (2.0 * areas * np.linalg.norm(self.up)))
                component_diffuse = pvlib.irradiance.isotropic(
                    np.degrees(tilt), diffuse_intensity)
                component_fd[ci] += component_diffuse
            # assert not any(component_fd[ci] == 0)

    def _ipython_display_(self):
        # This needs to actually display, which is not the same as
        # returning a display.
        cam = pythreejs.PerspectiveCamera(
            position=[25, 35, 100], fov=20,
            children=[pythreejs.AmbientLight()],
        )
        children = [cam, pythreejs.AmbientLight(color="#dddddd")]
        material = pythreejs.MeshBasicMaterial(
            color="#ff0000", vertexColors="VertexColors", side="DoubleSide"
        )
        for model in self.components:
            mesh = pythreejs.Mesh(
                geometry=model.geometry, material=material, position=[0, 0, 0]
            )
            children.append(mesh)

        scene = pythreejs.Scene(children=children)

        rendererCube = pythreejs.Renderer(
            camera=cam,
            background="white",
            background_opacity=1,
            scene=scene,
            controls=[pythreejs.OrbitControls(controlling=cam)],
            width=800,
            height=800,
        )

        return rendererCube


class PeriodicScene(Scene):
    r"""Container for a scene in which components are repeated.

    Args:
        period (np.ndarray, optional): Period of repetition for
            components in each direction.
        direction (np.ndarray, optional): Unit vectors for directions
            over which components will be repeated.
        count (np.ndarray, optional): Number of times that components
            will be repeated in each direction.
        dont_reflect (bool, optional): If True, the components will
            only be repeated in the provided directions. If False, the
            components will be repeated in both the provided directions
            and the reflections of those directions.
        dont_center (bool, optional): If False, the periodic components
            will be added such that the original components remain as
            close to the center of the scene as possible without modifying
            their original positions. If True, the periodic components
            will all be added in the provided directions.
        buffer_as_primary (bool, optional): If True, hits of periodic
            components will be treated as the original components during
            ray tracing.

    """

    period = traittypes.Array(
        np.zeros((3,), "f4")
    ).valid(check_dtype("f4"), check_shape(3))
    direction = traittypes.Array(np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    ], "f4")).valid(check_dtype("f4"), check_shape(3, 3))
    count = traittypes.Array(np.array([0, 0, 0], "i4")).valid(
        check_dtype("i4"), check_shape(3))
    dont_reflect = traitlets.Bool(False)
    dont_center = traitlets.Bool(False)
    buffer_as_primary = traitlets.Bool(False)

    def __init__(self, *args, **kwargs):
        self._buffer_meshes = []
        super(PeriodicScene, self).__init__(*args, **kwargs)

    @property
    def nperiodic_copies(self):
        r"""int: Number of periodic copies of each component in the
        scene."""
        out = self.periodic_shifts.shape[0]
        if self.dont_reflect:
            out_exp = np.prod(self.count) - 1
        else:
            out_exp = np.prod(2 * self.count + 1) - 1
        assert out == out_exp
        return out

    @property
    def ncomponents_periodic(self):
        r"""int: Total number of periodic components in the scene."""
        ncomponents = self.ncomponents
        return ncomponents * self.nperiodic_copies

    @property
    def transmittance_periodic(self):
        r"""list: Tranmittance values for each component's faces
        including periodic components."""
        out = []
        for c in self.components:
            out += [
                c.transmittance for _ in range(self.nperiodic_copies + 1)
            ]
        return out

    @property
    def reflectance_periodic(self):
        r"""list: Reflectance values for each component's faces
        including periodic components."""
        out = []
        for c in self.components:
            out += [
                c.reflectance for _ in range(self.nperiodic_copies + 1)
            ]
        return out

    @classmethod
    def get_periodic_shifts(cls, period, direction, count,
                            dont_reflect=False, dont_center=False):
        r"""Get the shifts that should be applied to plants.

        Args:
            period (np.ndarray): Length of the period along each
                direction.
            direction (np.ndarray): Unit vector for the directions
                along which the period should be applied.
            count (np.ndarray): Number of times the period should be
                repeated in each direction.
            include_origin (bool, optional): If True, include the origin
                in the returned shifts.
            dont_reflect (bool, optional): If True, the shifts will only
                occur in the positive direction along each axis.
            dont_center (bool, optional): If True, this shifts will not
                be centered on the origin.

        Returns:
            np.ndarray: Shifts in each coordinate that should be applied.

        """
        import itertools
        shifts = []
        opts = []
        for axis in range(3):
            if period[axis] == 0:
                opts.append([0])
                continue
            if dont_reflect:
                count_lh = -(count[axis] // 2)
                count_rh = count[axis] + count_lh
            else:
                count_lh = -count[axis]
                count_rh = count[axis] + 1
            if dont_center:
                count_rh = count_rh - count_lh
                count_lh = 0
            opts.append(list(range(count_lh, count_rh)))
        for xbuffer, ybuffer, zbuffer in itertools.product(*opts):
            if xbuffer == 0 and ybuffer == 0 and zbuffer == 0:
                continue

            def _shift(ibuffer, axis):
                return ibuffer * period[axis] * direction[axis, :]

            ishift = (
                _shift(xbuffer, 0)
                + _shift(ybuffer, 1)
                + _shift(zbuffer, 2)
            )
            shifts.append(ishift)
        return np.vstack(shifts)

    def post_cast(self, query_type, output):
        r"""Finalize the results from running the ray tracer.

        Args:
            query_type (QueryType): Raytrace query type of the output.
            output (object): Raytracer result.

        Returns:
            object: Finalized raytracer result.

        """
        nperiodic = self.nperiodic_copies
        ids = None
        if isinstance(output, dict):
            ids = output["geomID"]
        elif query_type == QueryType.INTERSECT:
            # TODO This isn't actually what pyembree outputs
            ids = output
        else:
            return output
        for compID in range(len(self.components)):
            geomID = compID * (nperiodic + 1)
            ids[ids == geomID] = compID
            bufferID = (compID if self.buffer_as_primary else -1)
            for vgeomID in range(geomID + 1, geomID + nperiodic + 1):
                ids[ids == vgeomID] = bufferID
        return output

    @cached_property
    def periodic_shifts(self):
        r"""np.ndarray: Shifts for periodic components."""
        return self.get_periodic_shifts(
            self.period, self.direction, self.count,
            dont_reflect=self.dont_reflect,
            dont_center=self.dont_center,
        )

    def add_component(self, component):
        r"""Add a component to the scene.

        Args:
            component (hothouse.model.Model): 3D component.

        """
        super(PeriodicScene, self).add_component(component)
        for shift in self.periodic_shifts:
            self._buffer_meshes.append(
                TriangleMesh(self.embree_scene,
                             component.triangles + shift)
            )
