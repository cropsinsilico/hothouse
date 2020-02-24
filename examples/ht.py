import matplotlib.pyplot as plt
import hothouse
import hothouse.blaster as hb
import hothouse.scene as hs
import hothouse.sun_calc as sc
import numpy as np
from hothouse.datasets import PLANTS 

def triangle_hits(o):
    triangle_hit_counts = []
    
    for i, component in enumerate(s.components):
        triangle_hit_counts.append(np.bincount(o['primID'][o['geomID'] == i],
                                           minlength = component.triangles.shape[0]))

    count = 0
    #print("triangle_hit_counts[0]: ", triangle_hit_counts[0])
    #for element in triangle_hit_counts:
    #    for i in element:
    #        print("triangle_hit_counts[", count, "]: ", i)
    #        count += 1

    return triangle_hit_counts




center = np.array([0.0, -100.0, 200])
forward = np.array([0.0, 1.0, 0.0])
up = np.array([0.0, 0.0, 1.0])

Npix = 1024

width = 500
height = 500

#sun information
latitude = 22.9068 #degrees
longitude = 43.1729 #degrees

#latitude = 46.77 #(in degrees)
#longitude = 117.2 #(in degrees)
standard_meridian = 120.0 #(in degrees)
day_of_year = 181 #equivalent to June 30
solar_noon = 12

hours_before_decimal = "9:45"
(hh, mm) = hours_before_decimal.split(':')
hour_of_day = int(hh) + (int(mm) * (1 / 60))
#print("hour_of_day: ", hour_of_day)

#rotation
theta = np.radians(30)
# in normal coordinates; y here
y = np.array(( (np.cos(theta), -np.sin(theta), 0),
               (np.sin(theta), np.cos(theta), 0),
               (0, 0, 1) ))
x = np.array(( (1, 0, 0),
               (0, np.cos(theta), -np.sin(theta)),
               (0, np.sin(theta), np.cos(theta)) ))
#y in normal coordinates; z here
z = np.array(( (np.cos(theta), 0, np.sin(theta)),
               (0, 1, 0),
               (-np.sin(theta), 0, np.cos(theta)) ))
print(x)
print(y)
print(z)

rotateX = x.dot(forward)
rotateY = y.dot(forward)
rotateZ = z.dot(forward)
print(rotateX)
print(rotateY)
print(rotateZ)


sc.sun_calcs(latitude, longitude, standard_meridian, day_of_year, hour_of_day)

#rb = hb.OrthographicRayBlaster(center, forward, up, width, height, Npix, Npix)
rb = hb.OrthographicRayBlaster(center, rotateX, up, width, height, Npix, Npix)

fname = PLANTS.fetch('fullSoy_2-12a.ply')
#fname = PLANTS.fetch('ambientFieldPly_10-29a.0001.ply')
#fname = 'ambientFieldPly_10-29a.0001.ply'
#fname = 'ambientFieldPly_10-29a.0100.ply'
p1 = hothouse.plant_model.PlantModel.from_ply(fname)
#p2 = hothouse.plant_model.PlantModel.from_ply(fname, origin = (100,100,0))

s = hs.Scene()
s.add_component(p1)
#s.add_component(p2)

N = 1
import time
t1 = time.time()
for i in range(N):
    o = rb.cast_once(s)
#o = rb.cast_once(s, True)
t2 = time.time()
print("Each takes {} seconds".format((t2 - t1) / N))

#triangle_hits(o)

#If a ray hits the instance, the geomID and primID members of the hit are set to the geometry ID and primitive ID of the hit primitive in the instanced scene, and the instID member of the hit is set to the geometry ID of the instance in the top-level scene.

#plt.plot(triangle_hit_counts[0])
#plt.plot(triangle_hit_counts[1])
#plt.savefig("hi_1.png")

o = {'tfar': o}
print(o)

#primIDs = o['tfar']['primID']
#print(primIDs)

#each triangle that gets hit by a ray at least once
#unique_primIDs = np.unique(o['tfar']['primID'])
#print(unique_primIDs)
#for i in unique_primIDs:
    #print(i)

plt.clf()
o['tfar'][o['tfar'] > 1e36] = np.nan
plt.imshow(o['tfar'].reshape((Npix, Npix), order='F'), origin='lower')
plt.colorbar()
plt.savefig("hi.png")
