import traittypes
import traitlets
import pythreejs
import numpy as np
import pvlib
from IPython.core.display import display

from .model import Model
from .blaster import RayBlaster, OrthographicRayBlaster, SunRayBlaster
from .traits_support import check_shape, check_dtype

from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh


def _get_cmap_texture(cmap_name):
    import matplotlib.cm as mcm

    values = np.array(mcm.get_cmap(cmap_name).colors)
    new_values = np.empty((256, 1, 4), dtype="u1")
    new_values[:, 0, :3] = (values * 255).astype("uint8")
    new_values[:, 0, 3] = 255
    return pythreejs.BaseDataTexture(data=new_values)


class Scene(traitlets.HasTraits):

    ground = traittypes.Array(np.array([0.0, 0.0, 0.0], "f4")).valid(
        check_dtype("f4"), check_shape(3)
    )
    up = traittypes.Array(np.array([0.0, 0.0, 1.0], "f4")).valid(
        check_dtype("f4"), check_shape(3)
    )
    north = traittypes.Array(np.array([0.0, 1.0, 0.0], "f4")).valid(
        check_dtype("f4"), check_shape(3)
    )
    components = traitlets.List(trait=traitlets.Instance(Model))
    blasters = traitlets.List(trait=traitlets.Instance(RayBlaster))
    meshes = traitlets.List(trait=traitlets.Instance(TriangleMesh))
    embree_scene = traitlets.Instance(rtcs.EmbreeScene, args=tuple())

    # TODO: Add surface for ground so that reflection from ground
    # is taken into account

    def add_component(self, component):
        self.components = self.components + [component]  # Force traitlet update
        self.meshes.append(TriangleMesh(self.embree_scene, component.triangles))

    def compute_hit_count(self, blaster):
        output = blaster.compute_count(self)
        component_counts = {}
        for ci, component in enumerate(self.components):
            hits = output["primID"][output["geomID"] == ci]
            component_counts[ci] = np.bincount(
                hits[hits >= 0], minlength=component.triangles.shape[0]
            )
        return component_counts

    def get_sun_blaster(
        self, latitude, longitude, date, direct_ppfd=1.0, diffuse_ppfd=1.0, **kwargs
    ):
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
                max_distance2, np.max(np.sum((c.vertices - self.ground) ** 2, axis=1))
            )
        max_distance = np.sqrt(max_distance2)
        kwargs.setdefault("zenith", self.up * max_distance)
        kwargs.setdefault("width", 2 * max_distance)
        kwargs.setdefault("height", 2 * max_distance)
        kwargs.setdefault(
            "intensity", (direct_ppfd * kwargs["width"] * kwargs["height"])
        )
        kwargs.setdefault("diffuse_intensity", diffuse_ppfd)
        blaster = SunRayBlaster(
            latitude=latitude,
            longitude=longitude,
            date=date,
            ground=self.ground,
            north=self.north,
            **kwargs
        )
        return blaster

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
            any_hits = counts["primID"] >= 0
            for ci, component in enumerate(self.components):
                idx_hits = np.logical_and(counts["geomID"] == ci, any_hits)
                norms = component.normals
                areas = component.areas
                if isinstance(blaster, OrthographicRayBlaster):
                    component_counts = np.bincount(
                        counts["primID"][idx_hits],
                        minlength=component.triangles.shape[0],
                    )
                    aoi = np.arccos(
                        np.dot(norms, -blaster.forward)
                        / (2.0 * areas * np.linalg.norm(blaster.forward))
                    )
                    if any_direction:
                        aoi[aoi > np.pi / 2] -= np.pi
                    else:
                        aoi[aoi > np.pi / 2] = np.pi  # No contribution
                    component_fd[ci] += (
                        component_counts * blaster.ray_intensity * np.cos(aoi) / areas
                    )
                else:
                    # TODO: This loop can be removed if AOI is calculated
                    # for each intersection by embree (or callback)
                    for idx_ray in np.where(idx_hits)[0]:
                        idx_scene = output["primID"][i]
                        aoi = np.arccos(
                            np.dot(norms[idx_scene], -blaster.directions[idx_ray, :])
                            / (
                                2.0
                                * areas[idx_scene]
                                * np.linalg.norm(blaster.directions[idx_ray, :])
                            )
                        )
                        if any_direction:
                            aoi[aoi > np.pi / 2] -= np.pi
                        else:
                            aoi[aoi > np.pi / 2] = np.pi  # No contribution
                        component_fd[ci][idx_scene] += (
                            blaster.ray_intensity * np.cos(aoi) / areas[idx_scene]
                        )
                # Diffuse
                # TODO: This assumes diffuse light comes from everywhere
                tilt = np.arccos(
                    np.dot(norms, self.up) / (2.0 * areas * np.linalg.norm(self.up))
                )
                component_fd[ci] += pvlib.irradiance.isotropic(
                    np.degrees(tilt), blaster.diffuse_intensity
                )
        return component_fd

    def _ipython_display_(self):
        # This needs to actually display, which is not the same as returning a display.
        cam = pythreejs.PerspectiveCamera(
            position=[25, 35, 100], fov=20, children=[pythreejs.AmbientLight()],
        )
        children = [cam, pythreejs.AmbientLight(color="#dddddd")]
        material = pythreejs.MeshBasicMaterial(
            map=_get_cmap_texture("magma"), side="DoubleSide"
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
