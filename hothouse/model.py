import os
import numpy as np
import functools
import traittypes
import traitlets
from itertools import tee
from . import sun_calc
from .traits_support import (
    check_shape, check_dtype, dependent_default, dependent_property
)


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
    r"""Container for 3D geometries that can be added to scenes.

    Args:
        vertices (np.ndarray): Set of vertices used by the geometry.
        indices (np.ndarray): Indices of vertices comprising each face
            in the geometry.
        attributes (np.ndarray, optional): Attributes of each face in
            the geometry (e.g. color).
        triangles (np.ndarray, optional): Positions of vertices that
            make up each face.
        normals (np.ndarray, optional): Normal unit vectors for each
            face.
        vertex_normals (np.ndarray, optional): Normal unit vectors for
            each vertex.

    """
    vertices = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f8")
    )
    indices = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("i4")
    )
    triangles = traittypes.Array().valid(
        check_shape(None, 3, 3), check_dtype("f8")
    )
    attributes = traitlets.Dict(traitlets.Union(
        [traitlets.CFloat(), traittypes.Array()]
    ))
    normals = traittypes.Array().valid(
        check_shape(None, 3), check_dtype("f8")
    )
    vertex_normals = traittypes.Array(None, allow_none=True).valid(
        check_shape(None, 3, 3), check_dtype("f8")
    )
    # TODO: Wavelength dependence of reflectance/transmittance?

    # @traitlets.default("triangles")
    @dependent_default("triangles", ["indices", "vertices"], strict=True)
    def _default_triangles(self):
        if not (self.trait_has_value("indices")
                and self.trait_has_value("vertices")):
            raise traitlets.TraitError(
                "Either indices and vertices or triangles "
                "must be provided")
        return self.vertices[self.indices, :]

    @dependent_property("triangles")
    def triangles_f4(self):
        r"""np.ndarray: 32 bit version of face triangles."""
        return self.triangles.astype("f4")

    # @traitlets.default("indices")
    @dependent_default("indices", ["triangles"], strict=True)
    def _default_indices(self):
        return np.arange(
            self.triangles.shape[0] * 3, dtype="i4").reshape(
                self.triangles.shape[0], 3)

    @dependent_default("vertices", ["triangles"], strict=True)
    def _default_vertices(self):
        return self.triangles.reshape(
            (self.triangles.shape[0] * 3, 3)).astype("f8")

    @dependent_default("normals", ["triangles", "vertex_normals"])
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

    @property
    def nface(self):
        r"""int: Number of faces in the model."""
        if not self.trait_has_value("indices"):
            return self.triangles.shape[0]
        return self.indices.shape[0]

    @property
    def nvert(self):
        r"""int: Number of vertices in the model."""
        return self.vertices.shape[0]

    @property
    def _check_shape_nface(self):
        return check_shape(self.nface, ignore_trailing=True)

    @property
    def _check_shape_nvert(self):
        return check_shape(self.nvert, ignore_trailing=True)

    @traitlets.validate("attributes")
    def _validate_attributes(self, proposal):
        updates = {}
        for k, v in proposal['value'].items():
            if isinstance(v, float):
                v = v * np.ones(self.nface, "f8")
                updates[k] = v
            if k.startswith('vertex_'):
                self._check_shape_nvert(proposal['trait']._value_trait, v)
            else:
                self._check_shape_nface(proposal['trait']._value_trait, v)
            if k.endswith('colors'):
                check_shape(None, 3)(proposal['trait']._value_trait, v)
        if updates:
            proposal['value'].update(updates)
        return proposal['value']

    @traitlets.validate("normals", "indices", "triangles",
                        "vertex_normals")
    def _validate_nface(self, proposal):
        self._check_shape_nface(proposal['trait'], proposal['value'])
        return proposal['value']

    @traitlets.validate("vertices")
    def _validate_nvert(self, proposal):
        self._check_shape_nvert(proposal['trait'], proposal['value'])
        return proposal['value']

    @dependent_property("normals")
    def areas(self):
        r"""np.ndarray: Areas of the faces in this model."""
        return 0.5 * np.linalg.norm(self.normals, axis=1)

    @dependent_property("attributes", "face_colors")
    def vertex_colors(self):
        r"""np.ndarray: Colors of vertices in this model"""
        if "vertex_colors" in self.attributes:
            return self.attributes["vertex_colors"]
        if (((self.trait_has_value("face_colors")
              or "colors" in self.attributes)
             and self.face_colors is not None)):
            face_colors = self.face_colors
            count = np.zeros(self.nvert, "i4")
            out = np.zeros((self.nvert, 3), "i4")
            for face_color, face in zip(face_colors, self.indices):
                out[face, :] += face_color
                count[face] += 1
            count[count == 0] = 1
            return sun_calc.op_along_axis(
                np.divide, out, count, axis=0).astype("i4")
        return None

    @dependent_property("attributes", "vertex_colors")
    def face_colors(self):
        r"""np.ndarray: Colors of faces in this model."""
        if "colors" in self.attributes:
            return self.attributes["colors"]
        if (((self.trait_has_value("vertex_colors")
              or "vertex_colors" in self.attributes)
             and self.vertex_colors is not None)):
            vertex_colors = self.vertex_colors
            return vertex_colors[self.indices.view().flatten(), :].reshape(
                self.nface, -1, 3).mean(axis=1).astype("i4")
        return None

    @classmethod
    def from_file(cls, filename, **kwargs):
        r"""Create a model from a 3D mesh loaded from a file.

        Args:
            filename (str): Path to file.
            **kwargs: Additional keyword arguments are passed to either
                from_obj or from_ply after determine the file type from
                its extension.

        Returns:
            Model: Model containing the loaded geometry.

        Raises:
            ValueError: If filename has an unsupported extension.

        """
        ext = os.path.splitext(filename)[-1]
        if ext == '.obj':
            return cls.from_obj(filename, **kwargs)
        elif ext == '.ply':
            return cls.from_ply(filename, **kwargs)
        raise ValueError(f"Could not determine how to read the geometry "
                         f"from the file \"{filename}\" based on its "
                         f"extension")

    @classmethod
    def from_obj(cls, filename, attributes=None, **kwargs):
        r"""Create a model from a 3D mesh described by an ObjWavefront
        file.

        Args:
            filename (str): Path to ObjWavefront file.
            attributes (dict, optional): Mapping of attribute name to
                an array of values for that attribute at each face.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            Model: Model containing the loaded geometry.

        """
        import pywavefront
        obj = pywavefront.Wavefront(filename, collect_faces=True)
        if attributes is None:
            attributes = {}
        xyz_vert = np.asarray(obj.vertices, dtype='f8')
        xyz_faces = []
        vertex_normals = []
        # Not currently supported by pywavefront
        # if xyz_vert.shape[1] == 4 or xyz_vert.shape[1] == 7:
        #     attributes.setdefault('vertex_weights', xyz_vert[:, -1])
        #     xyz_vert = xyz_vert[:, :-1]
        if xyz_vert.shape[1] == 6:
            attributes.setdefault(
                "vertex_colors", 255 * xyz_vert[:, 3:].astype('i4'))
            xyz_vert = xyz_vert[:, :3]
        for mesh in obj.mesh_list:
            ifaces = np.asarray(mesh.faces, dtype='i4')
            xyz_faces.append(ifaces)
            for material in mesh.materials:
                if material.has_normals:
                    i0 = (material.has_uvs * 2)
                    vertex_data = np.asarray(
                        material.vertices, dtype='f8').reshape(
                            (-1, 3, material.vertex_size))
                    if material.has_normals:
                        vertex_normals.append(
                            vertex_data[:, :, i0:(i0 + 3)])
        xyz_faces = np.concatenate(xyz_faces, axis=0)
        if len(vertex_normals) > 0 and "vertex_normals" not in kwargs:
            kwargs['vertex_normals'] = np.concatenate(
                vertex_normals, axis=0)
        out = cls(
            vertices=xyz_vert,
            indices=xyz_faces,
            attributes=attributes,
            **kwargs
        )
        return out

    @classmethod
    def from_ply(cls, filename, attributes=None, **kwargs):
        r"""Create a model from a 3D mesh described by a ply file.

        Args:
            filename (str): Path to ply file.
            attributes (dict, optional): Mapping of attribute name to
                an array of values for that attribute at each face.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            Model: Model containing the loaded geometry.

        """
        # This is probably not the absolute best way to do this.
        from plyfile import PlyData
        plydata = PlyData.read(filename)
        if attributes is None:
            attributes = {}
        vertices = plydata["vertex"][:]
        faces = plydata["face"][:]
        xyz_faces = []
        for face in _ensure_triangulated(faces):
            indices = face[0]
            xyz_faces.append(indices)

        xyz_vert = np.stack([vertices[ax] for ax in "xyz"], axis=-1)
        xyz_faces = np.stack(xyz_faces)
        if (("diffuse_red" in vertices.dtype.names
             and "vertex_colors" not in attributes)):
            attributes["vertex_colors"] = np.stack(
                [vertices["diffuse_{}".format(c)]
                 for c in ("red", "green", "blue")],
                axis=-1,
            )
        obj = cls(
            vertices=xyz_vert.astype("f8"),
            indices=xyz_faces.astype('i4'),
            attributes=attributes,
            **kwargs
        )
        return obj

    @property
    def geometry(self):
        r"""pythreejs.BufferGeometry: Model geometry."""
        import pythreejs
        faces = self.indices.tolist()
        colors = None
        if self.vertex_colors is not None:
            colors = [
                f'#{r:02x}{g:02x}{b:02}'
                for r, g, b in self.vertex_colors.tolist()
            ]
        faces = [
            f + [None, ([colors[i] for i in f] if colors is not None
                        else None),
                 None]
            for f in faces
        ]
        kwargs = dict(
            vertices=self.vertices.astype("f4").tolist(),
            faces=faces,
        )
        # attributes = dict(
        #     position=pythreejs.BufferAttribute(
        #         self.vertices.astype("f4"), normalized=False),
        #     index=pythreejs.BufferAttribute(
        #         self.indices.ravel(order="C").astype("u4"),
        #         normalized=False
        #     ),
        # )
        if colors is not None:
            kwargs["colors"] = colors
            # attributes["color"] = pythreejs.BufferAttribute(colors)
            # Face colors requires
            # https://speakerdeck.com/yomotsu/low-level-apis-using-
            #     three-dot-js?slide=22
            # and
            # https://github.com/mrdoob/three.js/blob/master/src/
            #     renderers/shaders/ShaderLib.js
        geometry = pythreejs.Geometry(**kwargs)
        # geometry = pythreejs.BufferGeometry(attributes=attributes)
        geometry.exec_three_obj_method("computeFaceNormals")
        return geometry

    def translate(self, delta):
        r"""Shift the vertices in the model geometry.

        Args:
            delta (np.ndarray): Shift to apply.

        """
        if self.trait_metadata("vertices", "default_set") is False:
            self.vertices = self.vertices + delta
        else:
            self.triangles = self.triangles + delta

    # def rotate(self, q, origin="barycentric"):
    #     """
    #     This expects a quaternion as input.
    #     """
    #     pass
