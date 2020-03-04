import matplotlib.pyplot as plt
import hothouse
import numpy as np
from hothouse.datasets import PLANTS

center = np.array([0.0, -100.0, 200])
forward = np.array([0.0, 1.0, 0.0])
up = np.array([0.0, 0.0, 1.0])

Npix = 1024

width = 400
height = 400

rb = hothouse.OrthographicRayBlaster(center, forward, up,
                                     width, height, Npix, Npix)

fname = PLANTS.fetch('fullSoy_2-12a.ply')
p1 = hothouse.PlantModel.from_ply(fname)
p2 = hothouse.PlantModel.from_ply(fname, origin = (100,100,0))

s = hothouse.Scene()
s.add_component(p1)
s.add_component(p2)

o = rb.cast_once(s)
o[o > 1e35] = np.nan

plt.clf()
plt.imshow(o.reshape((Npix, Npix), order='F'), origin='lower')
plt.colorbar()
plt.savefig("distance.png")
