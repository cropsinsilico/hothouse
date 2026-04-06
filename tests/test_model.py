import pytest
import os
import numpy as np
import traitlets
from hothouse import model


def test_untriangulated_ply(datadir):
    r"""Test for creating a mesh from a Ply file that includes faces
    with more than 3 vertices."""
    fname = os.path.join(datadir, 'pyramid_untriangulated.ply')
    instance = model.Model.from_ply(fname)
    assert instance.nface == 6
    assert instance.nvert == 5


@pytest.mark.parametrize("instance_type", ["triangles", "indices"],
                         scope="class")
class TestModel:
    r"""Tests for Model class."""

    cls = model.Model

    @pytest.fixture(scope="class")
    def skip_if_not_triangles(self, instance_type):
        if instance_type != "triangles":
            pytest.skip("Only enabled for first instance type")

    @pytest.fixture(scope="class")
    def nface(self):
        return 5

    @pytest.fixture(scope="class")
    def triangles(self, nface):
        return np.arange(nface * 3 * 3, dtype="f8").reshape(nface, 3, 3)

    @pytest.fixture(scope="class")
    def vertices(self, triangles, nface):
        return triangles.reshape(nface * 3, 3)

    @pytest.fixture(scope="class")
    def indices(self, nface):
        return np.arange(nface * 3, dtype="i4").reshape(nface, 3)

    @pytest.fixture(scope="class")
    def vertex_colors(self, nface):
        out = np.zeros((nface * 3, 3), "i4")
        out[-3:, :] = [255, 0, 0]
        return out

    @pytest.fixture(scope="class")
    def face_colors(self, nface):
        out = np.zeros((nface, 3), "i4")
        out[-1, :] = [255, 0, 0]
        return out

    @pytest.fixture(scope="class")
    def instance_triangles(self, triangles, face_colors):
        return self.cls(triangles=triangles,
                        attributes={"colors": face_colors})

    @pytest.fixture(scope="class")
    def instance_indices(self, indices, vertices, vertex_colors):
        return self.cls(vertices=vertices, indices=indices,
                        attributes={"vertex_colors": vertex_colors})

    @pytest.fixture(scope="class")
    def create_instance(self, instance_type,
                        triangles, indices, vertices,
                        vertex_colors, face_colors):

        def _create_instance(no_colors=False):
            if instance_type == 'triangles':
                attributes = (
                    {} if no_colors
                    else {"colors": face_colors}
                )
                return self.cls(
                    triangles=triangles,
                    attributes=attributes,
                    vertex_normals=None,
                )
            else:
                attributes = (
                    {} if no_colors
                    else {"vertex_colors": vertex_colors}
                )
                return self.cls(
                    vertices=vertices, indices=indices,
                    attributes=attributes,
                    vertex_normals=None,
                )

        return _create_instance

    @pytest.fixture(scope="class")
    def instance(self, create_instance):
        return create_instance()

    @pytest.fixture
    def fresh_instance(self, create_instance):
        return create_instance(no_colors=True)

    def test_traits_errors(self, skip_if_not_triangles):
        r"""Test errors raised for invalid traits."""
        with pytest.raises(traitlets.TraitError):
            self.cls().triangles

    def test_attributes(self, nface, instance):
        r"""Test dependent instance properties."""
        assert instance.areas.shape == (nface, )
        assert instance.normals.shape == (nface, 3)
        assert instance.geometry

    def test_defaults(self, assert_allclose, instance,
                      triangles, vertices, indices,
                      face_colors, vertex_colors):
        r"""Test default triangles/indices/vertices."""
        assert_allclose(instance.triangles, triangles)
        assert_allclose(instance.indices, indices)
        assert_allclose(instance.vertices, vertices)
        assert_allclose(instance.vertices[instance.indices], triangles)
        assert_allclose(instance.face_colors, face_colors)
        assert_allclose(instance.vertex_colors, vertex_colors)

    @pytest.mark.parametrize("delta", [
        5.0,
        np.array([1.0, 2.0, 3.0], "f8"),
    ])
    def test_translate(self, assert_allclose, fresh_instance,
                       triangles, vertices, indices, delta):
        r"""Test model translation."""
        expected_indices = indices
        expected_triangles = triangles + delta
        expected_vertices = vertices + delta
        fresh_instance.translate(delta)
        assert_allclose(fresh_instance.triangles, expected_triangles)
        assert_allclose(fresh_instance.indices, expected_indices)
        assert_allclose(fresh_instance.vertices, expected_vertices)
        assert fresh_instance.face_colors is None
        assert fresh_instance.vertex_colors is None
        # Update to triangles
        fresh_instance.triangles = triangles
        assert_allclose(fresh_instance.triangles, triangles)
        assert_allclose(fresh_instance.indices, indices)
        assert_allclose(fresh_instance.vertices, vertices)
        # Update to vertices & indices
        with fresh_instance.hold_trait_notifications():
            fresh_instance.vertices = vertices
            fresh_instance.indices = indices
        assert_allclose(fresh_instance.triangles, triangles)
        assert_allclose(fresh_instance.indices, indices)
        assert_allclose(fresh_instance.vertices, vertices)

    def test_from_file(self, geometry_fname, skip_if_not_triangles):
        for ftype in ['ply', 'obj']:
            instance = self.cls.from_file(
                geometry_fname('pyramid', ftype))
            assert instance.geometry
        with pytest.raises(ValueError):
            self.cls.from_file('invalid.png')
