import pythreejs
from IPython.core.display import display
import matplotlib.cm as mcm
import numpy as np
import traitlets
import traittypes
from .scene import Scene
from .blaster import RayBlaster


def _get_cmap_texture(cmap_name):

    values = mcm.get_cmap(cmap_name)(np.mgrid[0.0:1.0:256j])
    values = (values * 255).astype("uint8").reshape((256, 1, 4))
    return pythreejs.BaseDataTexture(data=values)


class CropRenderer(traitlets.HasTraits):
    base_scene = traitlets.Instance(Scene, allow_none=False)
    scene = traitlets.Instance(pythreejs.Scene)
    colormap = traitlets.Unicode("magma")
    material = traitlets.Instance(pythreejs.MeshBasicMaterial)
    renderer = traitlets.Instance(pythreejs.Renderer)
    camera = traitlets.Instance(pythreejs.Camera)
    controls = traitlets.List()
    blaster = traitlets.Instance(RayBlaster)
    # arrow = traitlets.Instance()

    @traitlets.default("renderer")
    def _renderer_default(self):
        r = pythreejs.Renderer(
            scene=self.scene,
            camera=self.camera,
            background="white",
            background_opacity=1,
            controls=self.controls,
            width=500,
            height=500,
        )
        traitlets.link((self, "camera"), (r, "camera"))
        traitlets.link((self, "controls"), (r, "controls"))
        return r

    @traitlets.observe("colormap")
    def _colormap_change(self, change):
        self.material.map = _get_cmap_texture(change["new"])

    @traitlets.default("material")
    def _material_default(self):
        material = pythreejs.MeshBasicMaterial(
            map=_get_cmap_texture(self.colormap), side="DoubleSide"
        )
        return material

    @traitlets.default("camera")
    def _camera_default(self):
        cam = pythreejs.PerspectiveCamera(
            position=[25, 35, 100], fov=20, children=[pythreejs.AmbientLight()],
        )
        return cam

    @traitlets.default("scene")
    def _scene_default(self):
        children = [self.camera, pythreejs.AmbientLight(color="#dddddd")]

        for model in self.base_scene.components:
            children.append(
                pythreejs.Mesh(geometry=model.geometry, material=self.material)
            )
        return pythreejs.Scene(children=children)

    @traitlets.default("controls")
    def _controls_default(self):
        return [pythreejs.OrbitControls(controlling=self.camera)]

    def update_blaster(self, change=None):
        counts = self.base_scene.compute_hit_count(self.blaster)
        max_val = max(_.max() for _ in counts.values())
        for i, count in sorted(counts.items()):
            nv = self.base_scene.components[i].values * 0.0
            c = count / max_val
            nv[0::3, 0] = c
            nv[1::3, 0] = c
            nv[2::3, 0] = c
            self.base_scene.components[i].values = nv

    @traitlets.observe("blaster")
    def _blaster_changed(self, change):
        # Disconnect our old blaster
        if isinstance(change["old"], RayBlaster):
            change["old"].unobserve(self.update_blaster, traitlets.All)
        if not isinstance(change["new"], RayBlaster):
            return
        change["new"].observe(
            self.update_blaster,
            ["directions", "origins", "intensity", "diffuse_intensity"],
        )
        self.update_blaster()

    @traitlets.default("arrow")
    def _arrows_default(self, change):
        pass
