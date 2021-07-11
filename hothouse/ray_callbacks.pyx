# cython: language=c++

cimport numpy as np
import numpy as np
from pyembree.callback_handler cimport \
    RayCollisionCallback, CALLBACK_TERMINATE, CALLBACK_CONTINUE
from pyembree.rtcore_geometry cimport RTC_INVALID_GEOMETRY_ID
from pyembree.rtcore_ray cimport RTCRay

cdef class RayCollisionPrinter(RayCollisionCallback):
    cdef int callback(self, RTCRay &ray):
        print("Hi!", ray.geomID)
        return CALLBACK_TERMINATE


cdef class RayCollisionMultiBounce(RayCollisionCallback):

    cdef int iray
    cdef int nray
    cdef int maxbounce
    cdef public np.int32_t[:] nbounce
    cdef public np.int32_t[:,:] primID
    cdef public np.int32_t[:,:] geomID
    cdef public np.int32_t[:,:] instID
    cdef public np.float32_t[:,:] tfars
    cdef public np.float32_t[:,:,:] Ng
    cdef public np.float32_t[:,:,:] ray_dir
    cdef public np.float32_t[:,:] power

    def __cinit__(self, int nray, int maxbounce):
        self.iray = 0
        self.nray = nray
        self.maxbounce = maxbounce
        self.nbounce = np.zeros(nray, dtype="int32")
        self.primID = -1 * np.ones((nray, maxbounce), dtype="int32")
        self.geomID = -1 * np.ones((nray, maxbounce), dtype="int32")
        self.instID = -1 * np.ones((nray, maxbounce), dtype="int32")
        self.tfars = np.empty((nray, maxbounce), dtype="float32")
        self.Ng = np.empty((nray, maxbounce, 3), dtype="float32")
        self.ray_dir = np.empty((nray, maxbounce, 3), dtype="float32")
        self.power = np.ones((nray, maxbounce), dtype="float32")

    cdef void redirect_ray(self, RTCRay &ray):
        cdef int i
        cdef float nu, ux, uy, uz
        cdef float new_dir[3]
	# Move origin to point of intersection
        for i in range(3):
            ray.org[i] = ray.org[i] + ray.dir[i] * ray.tfar
        nu = np.sqrt(ray.Ng[0] * ray.Ng[0]
                     + ray.Ng[1] * ray.Ng[1]
                     + ray.Ng[2] * ray.Ng[2])
        nx = ray.Ng[0] / nu
        ny = ray.Ng[1] / nu
        nz = ray.Ng[2] / nu
	# Rotate inverse of original direction 180 deg around surface
	# normal to get new direction
        new_dir[0] = -(ray.dir[0] * (-1.0 + (2.0 * nx * nx))
                       + ray.dir[1] * (2.0 * nx * ny)
                       + ray.dir[2] * (2.0 * nx * nz))
        new_dir[1] = -(ray.dir[0] * (2.0 * ny * nx)
                       + ray.dir[1] * (-1.0 + (2.0 * ny * ny))
                       + ray.dir[2] * (2.0 * ny * nz))
        new_dir[2] = -(ray.dir[0] * (2.0 * nz * nx)
                       + ray.dir[1] * (2.0 * nz * ny)
                       + ray.dir[2] * (-1.0 + (2.0 * nz * nz)))
        for i in range(3):
            ray.dir[i] = new_dir[i]
            # Offset origin slightly along new path to prevent
            # intersecting at new origin
            ray.org[i] = ray.org[i] + 1000.0 * ray.dir[i] * np.finfo(np.float32).eps

    @property
    def bounces(self):
        out = {'nbounce': np.asarray(self.nbounce),
               'primID': np.asarray(self.primID),
               'geomID': np.asarray(self.geomID),
               'instID': np.asarray(self.instID),
               'tfars': np.asarray(self.tfars),
               'Ng': np.asarray(self.Ng),
               'ray_dir': np.asarray(self.ray_dir),
               'power': np.asarray(self.power)}
        return out

    cdef int callback(self, RTCRay &ray):
        cdef int idx
        if ray.geomID == RTC_INVALID_GEOMETRY_ID:
            self.iray += 1
            return CALLBACK_TERMINATE
        if self.nbounce[self.iray] >= self.maxbounce:
            self.iray += 1
            return CALLBACK_TERMINATE
        # print("Hi!", self.iray, ray.geomID, ray.tnear, ray.tfar, ray.primID, ray.org, ray.dir)
        # Record ray parameters for bounce
        idx = self.nbounce[self.iray]
        self.nbounce[self.iray] += 1
        self.primID[self.iray, idx] = ray.primID
        self.geomID[self.iray, idx] = ray.geomID
        self.instID[self.iray, idx] = ray.instID
        self.tfars[self.iray, idx] = ray.tfar
        for i in range(3):
            self.Ng[self.iray, idx, i] = ray.Ng[i]
            self.ray_dir[self.iray, idx, i] = ray.dir[i]
        # TODO: Update power based on transmittance/reflectance
        # Redirect ray
        self.redirect_ray(ray)
        # Reset ray parameters
        ray.tnear = 0.0
        ray.tfar = 1e37
        ray.geomID = RTC_INVALID_GEOMETRY_ID
        ray.primID = RTC_INVALID_GEOMETRY_ID
        ray.instID = RTC_INVALID_GEOMETRY_ID
        return CALLBACK_CONTINUE
