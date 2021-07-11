import numpy as np
import functools
import traittypes
import traitlets
from itertools import tee
import pythreejs

from plyfile import PlyData, PlyElement

from .traits_support import check_shape, check_dtype

from yggdrasil.communication import open_file_comm

cached_property = getattr(functools, "cached_property", property)

# From itertools cookbook
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _ensure_triangulated(faces):
    for face in faces:
        if len(face[0]) == 3:
            yield face
            continue
        # We are going to make the assumption that the face is convex
        # We choose the first vertex as our fan source
        indices, *rest = face
        base = indices[0]
        for pair in pairwise(indices[1:]):
            yield [np.array((base,) + pair)] + rest


class Model(traitlets.HasTraits):
    origin = traittypes.Array(None, allow_none=True).valid(
        check_shape(3), check_dtype("f4")
    )
    vertices = traittypes.Array(None, allow_none=True).valid(
        check_shape(None, 3), check_dtype("f4")
    )
    indices = traittypes.Array(None, allow_none=True).valid(
        check_shape(None, 3), check_dtype("i4")
    )
    attributes = traittypes.Array(None, allow_none=True)
    triangles = traittypes.Array(None, allow_none=True).valid(
        check_shape(None, 3, 3), check_dtype("f4")
    )
    normals = traittypes.Array(None, allow_none=True).valid(
        check_shape(None, 3), check_dtype("f4")
    )
    vertex_normals = traittypes.Array(None, allow_none=True).valid(
        check_shape(None, 3, 3), check_dtype("f4")
    )
    # TODO: Wavelength dependence?
    transmittance = traittypes.Array(None, allow_none=True).valid(
        check_dtype("f4"))
    reflectance = traittypes.Array(None, allow_none=True).valid(
        check_dtype("f4"))

    @classmethod
    def from_obj(cls, filename, transmittance=None, reflectance=None):
        # with open_file_comm(filename, 'r', filetype='obj') as comm:
        #     flag, obj = comm.recv()
        #     assert(flag)
        # xyz_vert = np.asarray([[v[k] for k in ['x', 'y', 'z']]
        #                        for v in obj['vertices']],
        #                       dtype='f4')
        # xyz_faces = np.asarray([[v['vertex_index'] for v in f]
        #                         for f in obj['faces']], dtype='i4')
        # triangles = np.asarray(obj.mesh, dtype='f4')
        # vertex_normals = obj.vertex_normals
        # if vertex_normals is not None:
        #     vertex_normals = np.asarray(vertex_normals, dtype='f4')
        import pywavefront
        obj = pywavefront.Wavefront(filename, collect_faces=True)
        xyz_vert = np.asarray(obj.vertices, dtype='f4')
        xyz_faces = []
        triangles = []
        vertex_normals = []
        colors = []
        for mesh in obj.mesh_list:
            ifaces = np.asarray(mesh.faces, dtype='i4')
            xyz_faces.append(ifaces)
            for material in mesh.materials:
                if material.has_normals:
                    vertex_data = np.asarray(
                        material.vertices, dtype='f4').reshape(
                            (-1, 3, material.vertex_size))
                    i0 = (material.has_uvs * 2)
                    i1 = (material.has_uvs * 2
                          + material.has_normals * 3)
                    i2 = (material.has_uvs * 2
                          + material.has_normals * 3
                          + material.has_colors * 3)
                    vertex_normals.append(
                        vertex_data[:, :, i0:(i0 + 3)])
                    colors.append(
                        vertex_data[:, :, i1:(i1 + 3)])
                    triangles.append(
                        vertex_data[:, :, i2:(i2 + 3)])
        xyz_faces = np.concatenate(xyz_faces, axis=0)
        triangles = np.concatenate(triangles, axis=0)
        print('triangles', np.min(xyz_vert, axis=0), np.max(xyz_vert, axis=0))
        vertex_normals = np.concatenate(vertex_normals, axis=0)
        colors = np.concatenate(colors, axis=0)
        if isinstance(transmittance, (float, np.float)):
            transmittance = transmittance * np.ones(triangles.shape[0], 'f4')
        if isinstance(reflectance, (float, np.float)):
            reflectance = reflectance * np.ones(triangles.shape[0], 'f4')
        out = cls(
            vertices=xyz_vert,
            indices=xyz_faces,
            # attributes=colors,
            triangles=triangles,
            vertex_normals=vertex_normals,
            transmittance=transmittance,
            reflectance=reflectance
        )
        return out

    @classmethod
    def from_ply(cls, filename, transmittance=None, reflectance=None):
        # This is probably not the absolute best way to do this.
        plydata = PlyData.read(filename)
        vertices = plydata["vertex"][:]
        faces = plydata["face"][:]
        triangles = []
        xyz_faces = []
        for face in _ensure_triangulated(faces):
            indices = face[0]
            vert = vertices[indices]
            triangles.append(np.array([vert["x"], vert["y"], vert["z"]]))
            xyz_faces.append(indices)

        xyz_vert = np.stack([vertices[ax] for ax in "xyz"], axis=-1)
        xyz_faces = np.stack(xyz_faces)
        colors = None
        if "diffuse_red" in vertices.dtype.names:
            colors = np.stack(
                [vertices["diffuse_{}".format(c)] for c in ("red", "green", "blue")],
                axis=-1,
            )
        triangles = np.array(triangles).swapaxes(1, 2)
        if isinstance(transmittance, (float, np.float)):
            transmittance = transmittance * np.ones(triangles.shape[0], 'f4')
        if isinstance(reflectance, (float, np.float)):
            reflectance = reflectance * np.ones(triangles.shape[0], 'f4')
        obj = cls(
            vertices=xyz_vert,
            indices=xyz_faces.astype('i4'),
            attributes=colors,
            triangles=triangles,
            transmittance=transmittance,
            reflectance=reflectance
        )

        return obj

    @property
    def geometry(self):
        attributes = dict(
            position=pythreejs.BufferAttribute(self.vertices, normalized=False),
            index=pythreejs.BufferAttribute(
                self.indices.ravel(order="C").astype("u4"), normalized=False
            ),
        )
        if self.attributes is not None:
            attributes["color"] = pythreejs.BufferAttribute(self.attributes)
            # Face colors requires
            # https://speakerdeck.com/yomotsu/low-level-apis-using-three-dot-js?slide=22
            # and
            # https://github.com/mrdoob/three.js/blob/master/src/renderers/shaders/ShaderLib.js
        geometry = pythreejs.BufferGeometry(attributes=attributes)
        geometry.exec_three_obj_method("computeFaceNormals")
        return geometry

    @traitlets.default("normals")
    def _default_normals(self):
        r"""Array of the normal vectors for the triangles in this model."""
        v10 = self.triangles[:, 1, :] - self.triangles[:, 0, :]
        v20 = self.triangles[:, 2, :] - self.triangles[:, 0, :]
        out = np.cross(v10, v20)
        if isinstance(self.vertex_normals, np.ndarray):
            # Direction from vertex normals
            mask = (np.einsum("ij, ij->i", out,
                              np.mean(self.vertex_normals, axis=1)) < 0)
            out[mask] *= -1
        return out

    @cached_property
    def areas(self):
        r"""Array of areas for the triangles in this model."""
        return 0.5 * np.linalg.norm(self.normals, axis=1)

    def translate(self, delta):
        self.vertices = self.vertices + delta

    def rotate(self, q, origin="barycentric"):
        """
        This expects a quaternion as input.
        """
        pass
