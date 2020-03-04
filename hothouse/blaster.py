import pyembree
import numpy as np

# pyembree receives origins and directions.


class RayBlaster:
    def __init__(self):
        pass

    def cast_once(self, scene, verbose_output = False):
        output = scene.embree_scene.run(
                                        self.origins, self.directions, query="DISTANCE", output=verbose_output
                                        #self.origins, self.directions, query="INTERSECT", output=verbose_output
        )
        return output

    def plot_distance(self, scene):
        output = self.cast_once()


class OrthographicRayBlaster(RayBlaster):
    def __init__(self, center, forward, up, width, height, nx, ny):
        # figure out the camera from this, including an up vector
        self.center = center
        self.forward = forward
        self.up = up
        self.east = np.cross(self.forward, self.up)
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        
        # here origin is not the center, but the bottom left
        self._directions = np.zeros((nx, ny, 3), dtype="f4")
        self._directions[:] = self.forward[None, None, :]
        self.directions = self._directions.view().reshape((nx * ny, 3))

        self._origins = np.zeros((nx, ny, 3), dtype="f4")
        offset_x, offset_y = np.mgrid[
            -width / 2 : width / 2 : nx * 1j, -height / 2 : height / 2 : ny * 1j
        ]
        self._origins[:] = (
            self.center
            + offset_x[..., None] * self.east
            + offset_y[..., None] * self.up
        )
        self.origins = self._origins.view().reshape((nx * ny, 3))

class ProjectionRayBlaster(RayBlaster):
    pass


class SphericalRayBlaster(RayBlaster):
    pass
