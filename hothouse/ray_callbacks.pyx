# cython: language=c++

cimport numpy as np
import numpy as np
from pyembree.callback_handler cimport \
    RayCollisionCallback, CALLBACK_TERMINATE, CALLBACK_CONTINUE
from pyembree.rtcore_geometry cimport RTC_INVALID_GEOMETRY_ID
from pyembree.rtcore_ray cimport RTCRayHit

cdef class RayCollisionPrinter(RayCollisionCallback):
    cdef int callback(self, RTCRayHit &ray):
        print("Hi!", ray.hit.geomID)
        return CALLBACK_TERMINATE

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
    cdef RTCRayHit* ray_queue
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
        self.ray_queue = <RTCRayHit*>PyMem_Malloc(self.maxbounce * sizeof(RTCRayHit))
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

    cdef float pop_ray(self, RTCRayHit &ray):
        if self.nq == 0:
            raise RuntimeError("No rays queued.")
        self.nq -= 1
        cdef int i
        ray.ray.dir_x = self.ray_queue[self.nq].ray.dir_x
        ray.ray.dir_y = self.ray_queue[self.nq].ray.dir_y
        ray.ray.dir_z = self.ray_queue[self.nq].ray.dir_z
        
	# Offset slightly to prevent intersection at orgin
        ray.ray.org_x = self.ray_queue[self.nq].ray.org_x + 1000.0 * ray.ray.dir_x * np.finfo(np.float32).eps
        ray.ray.org_y = self.ray_queue[self.nq].ray.org_y + 1000.0 * ray.ray.dir_y * np.finfo(np.float32).eps
        ray.ray.org_z = self.ray_queue[self.nq].ray.org_z + 1000.0 * ray.ray.dir_z * np.finfo(np.float32).eps
        
        ray.ray.tnear = self.ray_queue[self.nq].ray.tnear
        ray.ray.tfar = self.ray_queue[self.nq].ray.tfar
        ray.hit.geomID = self.ray_queue[self.nq].rayhit.geomID
        ray.hit.primID = self.ray_queue[self.nq].rayhit.primID
        ray.hit.instID[0] = self.ray_queue[self.nq].rayhit.instID[0]
        ray.ray.mask = self.ray_queue[self.nq].ray.mask
        ray.ray.time = self.ray_queue[self.nq].ray.time
        return self.power_queue[self.nq]

    cdef void reflect_ray(self, RTCRayHit &ray, float power):
        # TODO: Reflect more that one ray to account for spectral
        # dependency or subsurface reflection
        cdef int i
        cdef float nu, ux, uy, uz
        cdef float new_org[3], new_dir[3]
	# Move origin to point of intersection
        new_org[0] = ray.ray.org_x + ray.ray.dir_x * ray.ray.tfar
        new_org[1] = ray.ray.org_y + ray.ray.dir_y * ray.ray.tfar
        new_org[2] = ray.ray.org_z + ray.ray.dir_z * ray.ray.tfar
        nu = np.sqrt(ray.hit.Ng_x * ray.hit.Ng_x
                     + ray.hit.Ng_y * ray.hit.Ng_y
                     + ray.hit.Ng_z * ray.hit.Ng_z)
        nx = ray.hit.Ng_x / nu
        ny = ray.hit.Ng_y / nu
        nz = ray.hit.Ng_z / nu
	# Rotate inverse of original direction 180 deg around surface
	# normal to get new direction
        new_dir[0] = -(ray.ray.dir_x * (-1.0 + (2.0 * nx * nx))
                       + ray.ray.dir_y * (2.0 * nx * ny)
                       + ray.ray.dir_z * (2.0 * nx * nz))
        new_dir[1] = -(ray.ray.dir_x * (2.0 * ny * nx)
                       + ray.ray.dir_y * (-1.0 + (2.0 * ny * ny))
                       + ray.ray.dir_z * (2.0 * ny * nz))
        new_dir[2] = -(ray.ray.dir_x * (2.0 * nz * nx)
                       + ray.ray.dir_y * (2.0 * nz * ny)
                       + ray.ray.dir_z * (-1.0 + (2.0 * nz * nz)))
        cdef float R = 0.0
        if self.reflectance[ray.hit.geomID] is not NULL:
            R = self.reflectance[ray.hit.geomID][ray.hit.primID]
        self.add_ray(new_org, new_dir, R * power)

    cdef void transmit_ray(self, RTCRayHit &ray, float power):
        # TODO: Refraction
        cdef int i
        cdef float new_org[3], new_dir[3]
	# Move origin to point of intersection
        new_org[0] = ray.ray.org_x + ray.ray.dir_x * ray.ray.tfar
        new_org[1] = ray.ray.org_y + ray.ray.dir_y * ray.ray.tfar
        new_org[2] = ray.ray.org_z + ray.ray.dir_z * ray.ray.tfar
        new_dir[0] = ray.ray.dir_x
        new_dir[1] = ray.ray.dir_y
        new_dir[2] = ray.ray.dir_z
        cdef float T = 0.0
        if self.transmittance[ray.hit.geomID] is not NULL:
            T = self.transmittance[ray.hit.geomID][ray.hit.primID]
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

    cdef int callback(self, RTCRayHit &ray):
        cdef int idx
        if ray.hit.geomID == RTC_INVALID_GEOMETRY_ID:
            if self.nq > 0:
                self.ipower = self.pop_ray(ray)
                return CALLBACK_CONTINUE
            else:
                self.iray += 1
                self.ipower = 1.0
                return CALLBACK_TERMINATE
        if self.nbounce[self.iray] >= (self.maxbounce - 1):
            # Reset queued since there isn't any more room and
            # move to next original ray
            self.nq = 0
            self.iray += 1
            self.ipower = 1.0
            return CALLBACK_TERMINATE
        # print("Hi!", self.iray, ray.hit.geomID, ray.ray.tnear, ray.ray.tfar, ray.hit.primID, [ray.ray.org_x, ray.ray.org_y, ray.ray.org_z], [ray.ray.dir_x, ray.ray.dir_y, ray.ray.dir_z])
        # Record ray parameters for bounce
        idx = self.nbounce[self.iray]
        self.nbounce[self.iray] += 1
        self.primID[self.iray, idx] = ray.hit.primID
        self.geomID[self.iray, idx] = ray.hit.geomID
        self.instID[self.iray, idx] = ray.hit.instID[0]
        self.tfars[self.iray, idx] = ray.ray.tfar
        self.Ng[self.iray, idx, 0] = ray.hit.Ng_x
        self.Ng[self.iray, idx, 1] = ray.hit.Ng_y
        self.Ng[self.iray, idx, 2] = ray.hit.Ng_z
        self.ray_dir[self.iray, idx, 0] = ray.ray.dir_x
        self.ray_dir[self.iray, idx, 1] = ray.ray.dir_y
        self.ray_dir[self.iray, idx, 2] = ray.ray.dir_z
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
            return CALLBACK_TERMINATE
        # Reset ray parameters from queue
        self.ipower = self.pop_ray(ray)
        return CALLBACK_CONTINUE
