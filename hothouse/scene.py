import traittypes
import traitlets
import pythreejs
import numpy as np
import functools
# from IPython.core.display import display

from .model import Model
from .blaster import (
    QueryType, RayBlaster, SunRayBlaster,
)
from .traits_support import check_shape, check_dtype
from . import sun_calc

from embreex.mesh_construction import TriangleMesh

# Use native embreex scene
from embreex import rtcore_scene as rtcs
EmbreeScene = rtcs.EmbreeScene

# Use local subclass of embreex scene w/ support for callbacks
# from .callback_handler import CallbackScene
# EmbreeScene = CallbackScene

cached_property = getattr(functools, "cached_property", property)


class CastAccumulator(traitlets.HasTraits):

    r"""Wrapper for function to accumulate ray trace results.

    Args:
        name (str): Name of accumulator.
        function (callable, optional): Function method that should be
            used to accumulate ray trace results. If not provided, the
            Scene class method with the name f'_accumulate_{name}' will
            be used. The function must take the following arguments::

                dst (np.ndarray): Destination array that values
                    should be added to.
                component (hothouse.model.Model): Scene component
                    that value is being calculated for.
                ray_dir (np.ndarray): Directions of rays.
                power (np.ndarray): Power of rays.
                primID (np.ndarray): Index of the component face that
                    each ray intersects within the component geometry
                    that contains it. -1 for no intersection.
                tfar (np.ndarray): The distance that each ray
                    traveled before intersecting a surface in the
                    scene.
                Ng (np.ndarray): Normal vector for the surface that
                    each ray intersected.
                u (np.ndarray): Projection of the ray up along the
                    surface that each ray intersected
                    (barycentric u coordinate of hit).
                v (np.ndarray): Projection of the ray east along the
                    surface that each ray intersected
                    (barycentric v coordinate of hit).

        dtype (np.dtype, optional): Data type of the result array that
            the accumulator expects.
        value (int, float, optional): Initial value that the result
            array should be populated with.
        no_multibounce (bool, optional): If True, don't accumulate for
            bounces.
        kws (dict, optional): Keyword arguments to pass to the
            accumulator function when it is called via the accumulate
            method.

    """

    _default_dtypes = {'count': np.dtype('i4')}
    _default_no_bounce = ['tfar']
    _default_values = {'tfar': 1e37}
    _accum_keys = ["ray_dir", "power", 'primID', 'tfar', 'Ng', 'u', 'v']

    name = traitlets.Unicode()
    function = traitlets.Callable()
    dtype = traitlets.Instance(np.dtype)
    value = traitlets.CFloat()
    no_multibounce = traitlets.Bool(False)
    kws = traitlets.Dict()

    @traitlets.default("dtype")
    def _default_dtype(self):
        return self._default_dtypes.get(self.name, np.dtype("f4"))

    @traitlets.default("value")
    def _default_value(self):
        return self._default_values.get(self.name, 0)

    @traitlets.default("no_multibounce")
    def _default_no_multibounce(self):
        return (self.name in self._default_no_bounce)

    @traitlets.default("function")
    def _default_function(self):
        return getattr(Scene, f'_accumulate_{self.name}')

    def accumulate(self, dst, ci, component, counts,
                   idx=None, idx_bounces=None, **kwargs):
        r"""Accumulate raytracer results for a component.

        Args:
            dst (dict): Mapping to contain accumulated results.
            ci (int): Component index.
            component (hothouse.model.Model): Scene component
                that value is being calculated for.
            counts (dict): Raytracer results.
            idx (np.ndarray, optional): Index of rays that hit the
                provided component.
            idx_bounces (np.ndarray, optional): Index of ray bounces
                that hit the provided component.
            **kwargs: Additional keyword arguments can be used to
                override or supplement values in counts.

        """
        dst.setdefault(self.name, {})
        if ci not in dst[self.name]:
            dst[self.name][ci] = np.empty(
                (component.triangles.shape[0], ), self.dtype)
            dst[self.name][ci].fill(self.value)
        if idx is None:
            idx = (counts['geomID'] == ci)
        args = []
        for k in self._accum_keys:
            if k in kwargs:
                v = kwargs[k]
            else:
                v = counts[k]
            args.append(v[idx, ...])
        self.function(dst[self.name][ci], component, *args, **self.kws)
        if "bounces" in counts and not self.no_multibounce:
            self.accumulate(dst, ci, component, counts["bounces"],
                            idx=idx_bounces)

    @classmethod
    def from_kwargs(cls, name, v):
        r"""Create a CastAccumulator instance from user provided keyword
        arguments.

        Args:
            name (str): Accumulator name.
            v (callable, tuple, dict): Callable accumulator function,
                tuple containing an accumulator function and a dictionary
                of keyword arguments that should be passed to the
                accumulator function, or a dictionary of keyword
                arguments to pass to the CastAccumulator constructor.

        Returns:
            CastAccumulator: Accumulator instance.

        """
        if v is True:
            return cls(name=name)
        if isinstance(v, tuple):
            return cls(name=name, function=v[0], kws=v[1])
        elif isinstance(v, dict):
            return cls(name=name, **v)
        elif isinstance(v, cls):
            return v
        return cls(name=name, function=v)


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

    def compute_count(self, blasters, accumulators=None,
                      any_direction=True, **kwargs):
        r"""Run the raytracer to determine how each ray from a set of
        blasters will intersect this scene.

        Args:
            blasters (hothouse.blaster.RayBlaster, list): One or more
                blasters to cast on the scene.
            accumulators (dict, optional): Mapping between property
                names and CastAccumulator or values that can be passed to
                CastAccumulator.from_kwargs to create an accumulator.
                Default accumulators (which can be turned off in the
                provided accumulators dictionary via False) include::

                    count (np.ndarray): Number of rays intersecting each
                        face.
                    flux (np.ndarray): Flux incident on each face.
                    tfar (np.ndarray): The smallest distance that a ray
                        from any of the blasters traveled before
                        intersecting the face (does not include
                        reflected/transmitted arrays).

            any_direction (bool, optional): If True, light is deposited
                on component reguardless of if the blaster rays hit the
                front or back of a component surface. If False, light
                is only deposited if the blaster rays hit the front.
                Defaults to True.
            **kwargs: Additional keyword arguments are passed to
                compute_count for each blaster.

        Returns:
            dict: Raytrace results for each accumulated property stored
                as a dictionary mapping between component index and
                results array for each face in the component.

        """
        if isinstance(blasters, RayBlaster):
            blasters = [blasters]
        if accumulators is None:
            accumulators = {}
        accumulators.setdefault('count', True)
        accumulators.setdefault('flux', True)
        accumulators.setdefault('tfar', True)
        accum = {}
        out = {}
        for k, v in accumulators.items():
            if v is False:
                continue
            if isinstance(v, CastAccumulator):
                accum[k] = v
            elif k == 'flux' and v is True:
                accum[k] = CastAccumulator.from_kwargs(
                    k, {'kws': {'any_direction': any_direction}})
            elif k == 'flux_density' and v is True:
                assert accumulators.get('flux', False)
                continue  # Calculate from flux
            else:
                accum[k] = CastAccumulator.from_kwargs(k, v)
        for blaster in blasters:
            counts = blaster.compute_count(self, **kwargs)
            for ci, component in enumerate(self.components):
                idx = (counts['geomID'] == ci)
                idx_bounces = (
                    None if "bounces" not in counts
                    else (counts['bounces']['geomID'] == ci)
                )
                for v in accum.values():
                    v.accumulate(out, ci, component, counts,
                                 idx=idx, idx_bounces=idx_bounces,
                                 ray_dir=blaster.directions,
                                 power=blaster.ray_intensity)
                if (('flux' in accum
                     and blaster.diffuse_intensity > 0)):
                    out['flux'][ci] += (
                        component.areas * blaster.diffuse_intensity
                        * sun_calc.incident_power_diffuse(
                            self.up, component.normals,
                            area=component.areas,
                        )
                    )
        if ((accumulators.get('flux_density', False)
             and 'flux_density' not in out)):
            out['flux_density'] = {
                ci: out['flux'][ci] / component.areas
                for ci, component in enumerate(self.components)
            }
        return out

    @classmethod
    def _accumulate_count(cls, dst, component, ray_dir, power, primID,
                          tfar, Ng, u, v):
        dst[:] += np.bincount(primID, minlength=dst.shape[0])

    @classmethod
    def _accumulate_flux(cls, dst, component, ray_dir, power, primID,
                         tfar, Ng, u, v, any_direction=True):
        incident_power = power * sun_calc.incident_power_direct(
            ray_dir, component.normals[primID], component.areas[primID],
            any_direction=any_direction)
        dst[:] += np.bincount(primID, weights=incident_power,
                              minlength=dst.shape[0])

    @classmethod
    def _accumulate_tfar(cls, dst, component, ray_dir, power, primID,
                         tfar, Ng, u, v):
        for iray, iprimID in enumerate(primID):
            if tfar[iray] < dst[iprimID]:
                dst[iprimID] = tfar[iray]

    def compute_hit_count(self, blaster, **kwargs):
        r"""Run the raytracer to determine how many rays will hit each
        component face in the scene.

        Args:
            blaster (hothouse.blaster.RayBlaster): Blaster containing
                rays to trace.
            **kwargs: Additional keyword arguments are passed to
                the compute_count method for the blaster.

        Returns:
            dict: Mapping between component index and arrays of hit
                counts for each face in the component geometry.

        """
        accumulators = {
            'count': True,
            'flux': False,
            'tfar': False,
        }
        out = self.compute_count(blaster, accumulators=accumulators,
                                 **kwargs)
        return out['count']

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
                           any_direction=True, multibounce=False,
                           power_threshold=0.001, **kwargs):
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
            multibounce (bool, optional): If True, rays should be tracked
                through reflections/transmission.
            power_threshold (float, optional): Threshold below which rays
                should no longer be tracked during bounces.
            **kwargs: Additional keyword arguments are passed to the
                SunRayBlaster constructor.

        Returns:
            dict: Mapping from scene component to an array of photon flux
                density values for each triangle in the component.

        """
        rb = self.get_sun_blaster(latitude, longitude, date,
                                  direct_ppfd=direct_ppfd,
                                  diffuse_ppfd=diffuse_ppfd, **kwargs)
        return self.compute_flux_density(rb, any_direction=any_direction,
                                         multibounce=multibounce,
                                         power_threshold=power_threshold)

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
            ppfd_tot = sun_calc.solar_ppfd(latitude, longitude, frame,
                                           altitude=altitude)
            sun = self.get_sun_blaster(latitude, longitude, frame,
                                       nx=nx, ny=ny,
                                       direct_ppfd=ppfd_tot['direct'],
                                       diffuse_ppfd=ppfd_tot['diffuse'])
            o = camera.compute_flux_density(self, sun, multibounce=True)
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

    def compute_flux(self, light_sources, **kwargs):
        r"""Compute the flux on each scene element from a set of light
        sources. Values will be calculated from the 'intensity'
        attribute of the light source blasters such that the flux will
        have the same units as intensity.

        Args:
            light_sources (list): Set of RayBlasters used to determine
                the light incident on scene elements.
            **kwargs: Additional keyword arguments are passed to
                the compute_count method.

        Returns:
            dict: Mapping from scene component to an array of flux
                values for each triangle in the component.

        """
        accumulators = {
            'flux': True,
            'count': False,
            'tfar': False,
        }
        out = self.compute_count(
            light_sources, accumulators=accumulators, **kwargs
        )
        return out['flux']

    def compute_flux_density(self, light_sources, **kwargs):
        r"""Compute the flux density on each scene element from a
        set of light sources. Values will be calculated from the
        'intensity' attribute of the light source blasters such that
        the flux density will have units of

            [intensity units] / [distance unit from scene] ** 2.

        Args:
            light_sources (list): Set of RayBlasters used to determine
                the light incident on scene elements.
            **kwargs: Additional keyword arguments are passed to
                the compute_count method.

        Returns:
            dict: Mapping from scene component to an array of flux
                density values for each triangle in the component.

        """
        accumulators = {
            'flux_density': True,
            'count': False,
            'tfar': False,
        }
        out = self.compute_count(
            light_sources, accumulators=accumulators, **kwargs
        )
        return out['flux_density']

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
