import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import datetime
import pytz
import hothouse
from hothouse.datasets import PLANTS
from hothouse.scene import Scene
from hothouse.blaster import OrthographicRayBlaster
from pvlib_model import sun_model


def get_scene(name):
    r"""Get a scene for the specified name."""
    if name == 'plant':
        ground = np.array([0.0, 0.0, 200.0], dtype='f4')
        fname = PLANTS.fetch('fullSoy_2-12a.ply')
        center = np.array([-25.0, -100.0, 500], dtype='f4')
        forward = np.array([0.25, 1.0, 0.0], dtype='f4')
        up = np.array([0.0, 0.0, 1.0], dtype='f4')
        width = height = 800
    elif name == 'sphere':
        ground = np.array([0.0, 0.0, -100.0], dtype='f4')
        fname = os.path.join('data', 'sphere.ply')
        center = np.array([0.0, -300.0, 0], dtype='f4')
        forward = np.array([0.0, 1.0, 0.0], dtype='f4')
        up = np.array([0.0, 0.0, 1.0], dtype='f4')
        # center = np.array([0.0, 0.0, 300], dtype='f4')
        # forward = np.array([0.0, 0.0, -1.0], dtype='f4')
        # up = np.array([0.0, 1.0, 0.0], dtype='f4')
        width = height = 400
    elif name == 'pyramid':
        ground = np.array([0.0, 0.0, 0.0], dtype='f4')
        fname = os.path.join('data', 'pyramid.ply')
        center = np.array([0.5, 0.5, 4.0], dtype='f4')
        forward = np.array([0.0, 0.0, -1.0], dtype='f4')
        up = np.array([0.0, 1.0, 0.0], dtype='f4')
        # center = np.array([0.5, -5.0, 0.8], dtype='f4')
        # forward = np.array([0.0, 1.0, 0.0], dtype='f4')
        # up = np.array([0.0, 0.0, 1.0], dtype='f4')
        width = height = 2
    p = hothouse.plant_model.PlantModel.from_ply(fname)
    scene = Scene(ground=ground)
    scene.add_component(p)
    # Camera
    # center = np.array([0.0, 0.0, 800], dtype='f4')
    # forward = np.array([0.0, 0.0, -1.0], dtype='f4')
    # up = np.array([0.0, 1.0, 0.0], dtype='f4')
    nx = ny = 1024
    camera = OrthographicRayBlaster(center=center, forward=forward,
                                    up=up, width=width, height=height,
                                    nx=nx, ny=ny)
    return scene, camera
        

def plot_light(scene, camera, latitude_deg, longitude_deg, date,
               fname="light.png", single_bounce=False):
    # Solar radiation model including atmosphere
    ppfd_tot = sun_model(latitude_deg, longitude_deg, date)  # W m-2

    # Blaster representing the sun
    nx = ny = 1024
    sun = scene.get_sun_blaster(latitude_deg, longitude_deg, date,
                                nx=nx, ny=ny,
                                direct_ppfd=ppfd_tot['direct'],
                                diffuse_ppfd=ppfd_tot['diffuse'],
                                multibounce=(not single_bounce))

    # Compute flux density on scene from sun
    o = camera.compute_flux_density(scene, sun)
    o[o <= 0] = np.nan

    # Plot result
    plt.clf()
    plt.imshow(o.reshape((camera.ny, camera.nx), order='F'),
               origin='lower', norm=LogNorm(50, 5.0e4))
    plt.colorbar()
    if fname is not None:
        fname = plt.savefig(fname)
    else:
        plt.show()


def iter_plot(scene, camera, latitude, longitude,
              t_start, t_stop, n_step, single_bounce=False):
    dates = pd.date_range(t_start, t_stop, periods=n_step)
    for date in dates:
        plot_light(scene, camera, latitude, longitude, date, fname=None,
                   single_bounce=single_bounce)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='plant',
                        choices=['plant', 'sphere', 'pyramid'])
    parser.add_argument('--iterate', action='store_true')
    parser.add_argument('--nsteps', default=10, type=int)
    # These default to the values for Champaign, IL
    # Sunrise: 2020-06-17  5:23:00
    # Sunset:  2020-06-17 19:25:00
    parser.add_argument('--latitude', default=40.1164, type=float)
    parser.add_argument('--longitude', default=-88.2434, type=float)
    parser.add_argument('--timezone', default='America/Chicago')
    parser.add_argument('--time', default='2020-06-17 5:23:00')
    parser.add_argument('--start-time', default='2020-06-17 5:23:00')
    parser.add_argument('--stop-time', default='2020-06-17 19:25:00')
    parser.add_argument('--single-bounce', action='store_true')
    args = parser.parse_args()

    scene, camera = get_scene(args.scene)
    timezone = pytz.timezone(args.timezone)
    for k in ['start_time', 'stop_time', 'time']:
        setattr(args, k, datetime.datetime.strptime(
            getattr(args, k), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone))

    if args.iterate:
        iter_plot(scene, camera, args.latitude, args.longitude,
                  args.start_time, args.stop_time, args.nsteps,
                  single_bounce=args.single_bounce)
    else:
        plot_light(scene, camera, args.latitude, args.longitude,
                   args.time, single_bounce=args.single_bounce)
        # , fname=('%s.png' % args.scene))
