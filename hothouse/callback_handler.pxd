# distutils: language=c++
# distutils: include_dirs = EMBREE_INCLUDE_DIR
cimport embreex.rtcore_ray as rtcr

cdef enum:
    CALLBACK_TERMINATE = 0
    CALLBACK_CONTINUE = 1

cdef class RayCollisionCallback:
    # The function callback needs to return either CALLBACK_TERMINATE or
    # CALLBACK_CONTINUE.  CALLBACK_CONTINUE will keep it running, but
    # assumes that you have done something to the ray.  Otherwise it will
    # enter into an endless loop.
    cdef int callback(self, rtcr.RTCRayHit &ray)

cdef class RayCollisionNull(RayCollisionCallback):
    pass
