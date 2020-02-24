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


center = np.array([0.0, 0.0, 500])
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

fname = PLANTS.fetch('fullSoy_2-12a.ply')
#fname = PLANTS.fetch('ambientFieldPly_10-29a.0001.ply')
#fname = 'ambientFieldPly_10-29a.0001.ply'
#fname = 'ambientFieldPly_10-29a.0100.ply'
p1 = hothouse.plant_model.PlantModel.from_ply(fname)
#p2 = hothouse.plant_model.PlantModel.from_ply(fname, origin = (100,100,0))

s = hs.Scene()
s.add_component(p1)
#s.add_component(p2)
sc.sun_calcs(latitude, longitude, standard_meridian, day_of_year, hour_of_day)

for rotation in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 359):
    forward = np.array([0.0, 0.0, -1.0])
    theta = np.radians(rotation)
    # in normal coordinates; y here
    up = np.array( [ np.cos(theta), np.sin(theta), 0.0] )
    #up = up / np.dot(up, up)
    print(rotation, up)

    #rb = hb.OrthographicRayBlaster(center, forward, up, width, height, Npix, Npix)
    rb = hb.OrthographicRayBlaster(center, forward, up, width, height, Npix, Npix)

    o = rb.cast_once(s)
    o = {'tfar': o}
    o['tfar'][o['tfar'] > 1e36] = np.nan
    plt.clf()
    plt.imshow(o['tfar'].reshape((Npix, Npix), order='F'), origin='lower')
    plt.colorbar()
    plt.savefig("hi-{0:03d}.png".format(rotation))
