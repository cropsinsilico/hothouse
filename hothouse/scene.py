from .model import Model

from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh


class Scene:
    components: [Model] = None
    meshes: [] = None

    def __init__(self):
        self.components = []
        self.meshes = []
        self.embree_scene = rtcs.EmbreeScene()

    def add_component(self, component):
        self.components.append(component)
        self.meshes.append(TriangleMesh(self.embree_scene, component.triangles))
