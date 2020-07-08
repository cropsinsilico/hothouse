# cython: language=c++

cimport numpy as np
import numpy as np
from pyembree.callback_handler cimport \
    RayCollisionCallback, _CALLBACK_TERMINATE, _CALLBACK_CONTINUE
from pyembree.rtcore_geometry cimport RTC_INVALID_GEOMETRY_ID
from pyembree.rtcore_ray cimport RTCRay

cdef class RayCollisionPrinter(RayCollisionCallback):
    cdef int callback(self, RTCRay &ray):
        print("Hi!", ray.geomID)
        return _CALLBACK_TERMINATE

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class RayCollisionMultiBounce(RayCollisionCallback):

    cdef int nray
    cdef int maxbounce
    cdef float power_threshold
    cdef int iray
    cdef float ipower
    cdef public np.int32_t[:] nbounce
    cdef public np.int32_t[:,:] primID
    cdef public np.int32_t[:,:] geomID
    cdef public np.int32_t[:,:] instID
    cdef public np.float32_t[:,:] tfars
    cdef public np.float32_t[:,:,:] Ng
    cdef public np.float32_t[:,:,:] ray_dir
    cdef public np.float32_t[:,:] power
    cdef int nq
    cdef RTCRay* ray_queue
    cdef float* power_queue
    cdef float** transmittance
    cdef float** reflectance

    def __cinit__(self, int nray, int maxbounce,
                  list transmittance, list reflectance):
        self.nray = nray
        self.maxbounce = maxbounce
        # TODO: Pass this in
        self.power_threshold = 0.001
        self.iray = 0
        self.ipower = 1.0
        self.nbounce = np.zeros(nray, dtype="int32")
        self.primID = -1 * np.ones((nray, maxbounce), dtype="int32")
        self.geomID = -1 * np.ones((nray, maxbounce), dtype="int32")
        self.instID = -1 * np.ones((nray, maxbounce), dtype="int32")
        self.tfars = np.empty((nray, maxbounce), dtype="float32")
        self.Ng = np.empty((nray, maxbounce, 3), dtype="float32")
        self.ray_dir = np.empty((nray, maxbounce, 3), dtype="float32")
        self.power = np.zeros((nray, maxbounce), dtype="float32")
        self.nq = 0
        self.ray_queue = <RTCRay*>PyMem_Malloc(self.maxbounce * sizeof(RTCRay))
        self.power_queue = <float*>PyMem_Malloc(self.maxbounce * sizeof(float))
        self.transmittance = <float**>PyMem_Malloc(len(transmittance) * sizeof(float*))
        self.reflectance = <float**>PyMem_Malloc(len(reflectance) * sizeof(float*))
        cdef i
        cdef np.ndarray[np.float32_t] buff
        for i in range(len(transmittance)):
            if isinstance(transmittance[i], np.ndarray):
                buff = np.ascontiguousarray(transmittance[i], dtype=np.float32)
                self.transmittance[i] = <float*> buff.data
            else:
                self.transmittance[i] = NULL
        for i in range(len(reflectance)):
            if isinstance(reflectance[i], np.ndarray):
                buff = np.ascontiguousarray(reflectance[i], dtype=np.float32)
                self.reflectance[i] = <float*> buff.data
            else:
                self.reflectance[i] = NULL

    def __dealloc__(self):
        PyMem_Free(self.ray_queue)
        PyMem_Free(self.power_queue)
        PyMem_Free(self.transmittance)
        PyMem_Free(self.reflectance)

    cdef int add_ray(self, float org[3], float dir[3],
                     float power) except -1:
        if power < self.power_threshold:
            return 0
        if self.nq >= self.maxbounce:
            raise RuntimeError("Too many rays queued.")
        cdef int i
        for i in range(3):
            self.ray_queue[self.nq].org[i] = org[i]
            self.ray_queue[self.nq].dir[i] = dir[i]
        self.ray_queue[self.nq].tnear = 0.0
        self.ray_queue[self.nq].tfar = 1e37
        self.ray_queue[self.nq].geomID = RTC_INVALID_GEOMETRY_ID
        self.ray_queue[self.nq].primID = RTC_INVALID_GEOMETRY_ID
        self.ray_queue[self.nq].instID = RTC_INVALID_GEOMETRY_ID
        self.ray_queue[self.nq].mask = -1
        self.ray_queue[self.nq].time = 0
        self.power_queue[self.nq] = power
        self.nq += 1
        return 0

    cdef float pop_ray(self, RTCRay &ray):
        if self.nq == 0:
            raise RuntimeError("No rays queued.")
        self.nq -= 1
        cdef int i
        for i in range(3):
            ray.org[i] = self.ray_queue[self.nq].org[i]
            ray.dir[i] = self.ray_queue[self.nq].dir[i]
	    # Offset slightly to prevent intersection at orgin
            ray.org[i] = ray.org[i] + 1000.0 * ray.dir[i] * np.finfo(np.float32).eps
        ray.tnear = self.ray_queue[self.nq].tnear
        ray.tfar = self.ray_queue[self.nq].tfar
        ray.geomID = self.ray_queue[self.nq].geomID
        ray.primID = self.ray_queue[self.nq].primID
        ray.instID = self.ray_queue[self.nq].instID
        ray.mask = self.ray_queue[self.nq].mask
        ray.time = self.ray_queue[self.nq].time
        return self.power_queue[self.nq]

    cdef void reflect_ray(self, RTCRay &ray, float power):
        # TODO: Reflect more that one ray to account for spectral
        # dependency or subsurface reflection
        cdef int i
        cdef float nu, ux, uy, uz
        cdef float new_org[3], new_dir[3]
	# Move origin to point of intersection
        for i in range(3):
            new_org[i] = ray.org[i] + ray.dir[i] * ray.tfar
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
        cdef float R = 0.0
        if self.reflectance[ray.geomID] is not NULL:
            R = self.reflectance[ray.geomID][ray.primID]
        self.add_ray(new_org, new_dir, R * power)

    cdef void transmit_ray(self, RTCRay &ray, float power):
        # TODO: Refraction
        cdef int i
        cdef float new_org[3], new_dir[3]
	# Move origin to point of intersection
        for i in range(3):
            new_org[i] = ray.org[i] + ray.dir[i] * ray.tfar
            new_dir[i] = ray.dir[i]
        cdef float T = 0.0
        if self.transmittance[ray.geomID] is not NULL:
            T = self.transmittance[ray.geomID][ray.primID]
        self.add_ray(new_org, new_dir, T * power)

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
            if self.nq > 0:
                self.ipower = self.pop_ray(ray)
                return _CALLBACK_CONTINUE
            else:
                self.iray += 1
                self.ipower = 1.0
                return _CALLBACK_TERMINATE
        if self.nbounce[self.iray] >= (self.maxbounce - 1):
            # Reset queued since there isn't any more room and
            # move to next original ray
            self.nq = 0
            self.iray += 1
            self.ipower = 1.0
            return _CALLBACK_TERMINATE
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
        self.power[self.iray, idx] = self.ipower
        # TODO: Update power based on transmittance/reflectance
        # Redirect ray(s)
        self.transmit_ray(ray, self.ipower)
        self.reflect_ray(ray, self.ipower)
        if self.nq == 0:
            # Terminate and move on if no rays meet power threshold
            # TODO: reset ray to ensure it is not logged twice
            self.iray += 1
            self.ipower = 1.0
            return _CALLBACK_TERMINATE
        # Reset ray parameters from queue
        self.ipower = self.pop_ray(ray)
        return _CALLBACK_CONTINUE
