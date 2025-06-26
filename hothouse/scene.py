import traittypes
import traitlets
import pythreejs
import numpy as np
import functools
import pvlib
# from IPython.core.display import display

from .model import Model
from .blaster import RayBlaster, OrthographicRayBlaster, SunRayBlaster
from .traits_support import check_shape, check_dtype
from .sun_calc import solar_ppfd

from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh

cached_property = getattr(functools, "cached_property", property)


class Scene(traitlets.HasTraits):

    ground = traittypes.Array(np.array([0.0, 0.0, 0.0], "f4")).valid(
        check_dtype("f4"), check_shape(3))
    up = traittypes.Array(np.array([0.0, 0.0, 1.0], "f4")).valid(
        check_dtype("f4"), check_shape(3))
    north = traittypes.Array(np.array([0.0, 1.0, 0.0], "f4")).valid(
        check_dtype("f4"), check_shape(3))
    components = traitlets.List(trait=traitlets.Instance(Model))
    blasters = traitlets.List(trait=traitlets.Instance(RayBlaster))
    meshes = traitlets.List(trait=traitlets.Instance(TriangleMesh))
    embree_scene = traitlets.Instance(rtcs.EmbreeScene, args=tuple())

    # TODO: Add surface for ground so that reflection from ground
    # is taken into account

    def post_cast(self, output):
        return output

    def add_component(self, component):
        # Force traitlet update
        self.components = self.components + [component]
        self.meshes.append(
            TriangleMesh(self.embree_scene, component.triangles))

    def compute_hit_count(self, blaster):
        output = blaster.compute_count(self)
        component_counts = {}
        for ci, component in enumerate(self.components):
            hits = output["primID"][output["geomID"] == ci]
            component_counts[ci] = np.bincount(
                hits[hits >= 0], minlength=component.triangles.shape[0]
            )
        return component_counts

    @property
    def transmittance(self):
        return [c.transmittance for c in self.components]

    @property
    def reflectance(self):
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

    def get_sun_blaster(self, latitude, longitude, date,
                        direct_ppfd=1.0, diffuse_ppfd=1.0, **kwargs):
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
                Earth for the specified location and time. Defaults
                to 1.0.
            diffuse_ppfd (float, optional): Diffuse Photosynthetic
                Photon Flux Density (PPFD) at the surface of the
                Earth for the specified location and time. Defaults
                to 1.0.

        Returns:
            SunRayBlaster: Blaster tuned to this scene.

        """
        # TODO: Calculate direct/diffuse ppfd from lat/long/date
        # using pvi if not provided
        max_distance2 = 0.0
        for c in self.components:
            max_distance2 = max(
                max_distance2,
                np.max(np.sum((c.vertices-self.ground)**2, axis=1)))
        max_distance = np.sqrt(max_distance2)
        kwargs.setdefault('zenith', self.up * max_distance + self.ground)
        kwargs.setdefault('diffuse_intensity', diffuse_ppfd)
        kwargs.setdefault('scene_limits', self.limits)
        blaster = SunRayBlaster(latitude=latitude,
                                longitude=longitude, date=date,
                                ground=self.ground, north=self.north,
                                **kwargs)
        blaster.intensity = direct_ppfd * blaster.width * blaster.height
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

    period = traittypes.Array(
        np.zeros((3,), "f4")
    ).valid(check_dtype("f4"), check_shape(3))
    direction = traittypes.Array(np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    ], "f4")).valid(check_dtype("f4"), check_shape(3, 3))
    count = traittypes.Array(np.array([1, 1, 1], "i4")).valid(
        check_dtype("i4"), check_shape(3))
    buffer_as_primary = traitlets.Bool(False)

    def __init__(self, *args, **kwargs):
        self._buffer_meshes = []
        self._buffer_component_map = {}
        self._buffer_component_triangles = []
        self._embree_scene = kwargs.get('embree_scene',
                                        rtcs.EmbreeScene())
        super(PeriodicScene, self).__init__(*args, **kwargs)

    @property
    def transmittance(self):
        out = super(PeriodicScene, self).transmittance
        self.add_component_buffers()
        for i, v in self._buffer_component_map.items():
            out += [self.components[i].transmittance for _ in v]
        return out

    @property
    def reflectance(self):
        out = super(PeriodicScene, self).reflectance
        self.add_component_buffers()
        for i, v in self._buffer_component_map.items():
            out += [self.components[i].reflectance for _ in v]
        return out

    @classmethod
    def get_periodic_shifts(cls, period, direction, count):
        import itertools
        shifts = []
        opts = []
        for axis in range(3):
            if period[axis] == 0:
                opts.append([0])
                continue
            else:
                opts.append(list(range(-count[axis], count[axis] + 1)))
        for xbuffer, ybuffer, zbuffer in itertools.product(*opts):
            if xbuffer == 0 and ybuffer == 0 and zbuffer == 0:
                continue

            def _shift(ibuffer, axis):
                return ibuffer * period[axis] * direction[axis, :]

            shifts.append(_shift(xbuffer, 0)
                          + _shift(ybuffer, 1)
                          + _shift(zbuffer, 2))
        return np.vstack(shifts)

    def post_cast(self, output):
        if self.buffer_as_primary:
            for ci, v in self._buffer_component_map.items():
                for ci_per in v:
                    output["geomID"][output["geomID"] == ci_per] = ci
            for irange, jrange in self._buffer_component_triangles[::-1]:
                output["primID"][output["primID"] >= jrange.start] -= (
                    jrange.start - irange.start)
        return output

    def add_component_buffers(self):
        if self._buffer_meshes:
            return
        shifts = self.get_periodic_shifts(self.period, self.direction,
                                          self.count)
        j = len(self.components)
        icount = 0
        jcount = sum([component.triangles.shape[0]
                      for component in self.components])
        for i, component in enumerate(self.components):
            irange = range(icount, icount + component.triangles.shape[0])
            self._buffer_component_map[i] = []
            for shift in shifts:
                jrange = range(jcount, jcount + component.triangles.shape[0])
                self._buffer_meshes.append(
                    TriangleMesh(self._embree_scene,
                                 component.triangles + shift)
                )
                self._buffer_component_map[i].append(j)
                self._buffer_component_triangles.append((irange, jrange))
                j += 1
                jcount += component.triangles.shape[0]
            icount += component.triangles.shape[0]

    @property
    def embree_scene(self):
        self.add_component_buffers()
        return self._embree_scene
