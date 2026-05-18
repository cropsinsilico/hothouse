import numpy as np
import pvlib

# FIX: take into account daylight savings time like in Example 11.1
#  ignoring for now

# FIX: figure out where this came from in old raytracer
kSOLAR_constant = 2600


def solar_ppfd(latitude, longitude, date, altitude=None, pressure=None,
               temperature=12.0, eta_par=0.368, eta_photon=4.56):
    r"""Determine the Photosynthetic Photon Flux Density (PPFD)
    received from the sun at a location and time.

    Args:
        latitude (float): Latitude in degrees.
        longitude (float): Longitude in degrees.
        date (datetime.datetime): Date & time w/ timezone information.
        altitude (float, optional): Altitude (in meters) used to compute
            solar position. If not provided, but pressure is, pressure
            will be used to calculate altitude.
        pressure (float, optional): Pressure (in Pa) used to compute
            solar position. If not provided, but altitude is, altitude
            will be used to calculate pressure.
        temperature (float, optional): Air temperature (in degrees C)
            used to compute solar position.
        eta_par (float, optional): Fraction of solar radiation (assuming
            black-body spectrum of 5800 K) that is photosynthetically
            active (wavelengths 400–700 nm).
        eta_photon (float, optional): Average number of photons per
            photosynthetically activate unit of radiation (in
            µmol s−1 W−1).

    Returns:
        dict: PPFD values for direct and diffuse light.

    """
    if pressure is None and altitude is None:
        pressure = 101325.0
        altitude = pvlib.atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = pvlib.atmosphere.alt2pres(altitude)
    elif altitude is None:
        altitude = pvlib.atmosphere.pres2alt(pressure)
    location = pvlib.location.Location(
        latitude, longitude, altitude=altitude, tz=date.tzinfo)
    date_pv = pvlib.tools._datetimelike_scalar_to_datetimeindex(date)
    solpos = pvlib.solarposition.get_solarposition(
        date_pv, latitude, longitude, pressure=pressure,
        temperature=temperature)
    dni_extra = pvlib.irradiance.get_extra_radiation(date_pv)
    airmass = pvlib.atmosphere.get_relative_airmass(
        solpos['apparent_zenith'])
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    tl = pvlib.clearsky.lookup_linke_turbidity(
        date_pv, latitude, longitude)
    cs = location.get_clearsky(
        date, solar_position=solpos, dni_extra=dni_extra,
        linke_turbidity=tl, airmass_absolute=am_abs,
    )
    # Convert from irradiance to PPFD
    irr2ppfd = eta_par * eta_photon
    ppfd = {'direct': irr2ppfd * cs['dni'].iloc[0],
            'diffuse': irr2ppfd * cs['dhi'].iloc[0]}
    return ppfd


def op_along_axis(op, a, b, axis=0):
    r"""Perform an operation between two matrices element by element
    that match along a specified axis. This is equivalent to performing
    the element-wise operation on a with an array that repeats b along
    the other dimensions of a to create an array of the same size as a.

    Args:
        op (np.ndarray): Function that performs an operation between
            two arrays of the same size elementwise.
        a (np.ndarray): First array in operation.
        b (np.ndarray): Second array in operation.
            number of elements along the specified axis.
        axis (int, optional): Dimension that element-wise operation.
            should be performed over.

    Returns:
        np.ndarray: Result of operation that is the same size as a.

    """
    return np.swapaxes(op(np.swapaxes(a, axis, -1), b), -1, axis)


def norm_along_axis(a, axis=0):
    r"""Normalize vectors along a given axis.

    Args:
        a (np.ndarray): Matrix to normalize.
        axis (int, optional): Axis that vectors occupy.

    Returns:
        np.ndarray: Normalized matrix with the same size as a.

    """
    norm = np.linalg.norm(a, axis=axis)
    if axis == 0:
        return np.divide(a, norm)
    return np.moveaxis(np.divide(np.moveaxis(a, axis, 0), norm), 0, axis)


def incident_power_direct(ray_dir, norm, area=None, any_direction=True):
    r"""Calculate the fraction of a ray flux that will be deposited on
    a face.

    Args:
        ray_dir (np.ndarray): Unit vectors for ray(s).
        norm (np.ndarray): Normal vectors for faces the ray(s) intersect.
        area (np.ndarray, None): Area of the faces the ray(s) intersect.
            This should also be half the magnitude of the normal vectors
            so passing it directly if already calculated can save time.
        any_direction (bool, optional): If True, power can be deposited
            from any direction. If False, power can only be deposited if
            a ray intersects a face from the direction of the face's
            normal vector.

    Returns:
        np.ndarray: Fraction of power deposited on each face (cos(theta))

    """
    if area is None:
        out = (
            np.sum(norm * -ray_dir, axis=-1)
            / (np.linalg.norm(norm, axis=-1) *
               np.linalg.norm(ray_dir, axis=-1))
        )
    else:
        out = (
            np.sum(norm * -ray_dir, axis=-1)
            / (2.0 * area * np.linalg.norm(ray_dir, axis=-1))
        )
    if any_direction:
        out = np.abs(out)
    else:
        out[out < 0] = 0
    return out


def incident_power_diffuse(up_dir, norm, area=None):
    r"""Calculate the fraction of a diffuse flux that will be deposited
    on a face.

    Args:
        up_dir (np.ndarray): Unit normal to the ground.
        norm (np.ndarray): Normal vectors for faces the ray(s) intersect.
        area (np.ndarray, None): Area of the faces the ray(s) intersect.
            This should also be half the magnitude of the normal vectors
            so passing it directly if already calculated can save time.

    Returns:
        np.ndarray: Fraction of diffuse flux deposited on each face.

    """
    if area is None:
        out = (
            np.sum(norm * up_dir, axis=-1)
            / (np.linalg.norm(norm, axis=-1) *
               np.linalg.norm(up_dir, axis=-1))
        )
    else:
        out = (
            np.sum(norm * up_dir, axis=-1)
            / (2.0 * area * np.linalg.norm(up_dir, axis=-1))
        )
    # return pvlib.irradiance.isotropic(
    #     np.degrees(stable_arccos(out)), 1.0)
    return (1 + out) / 2


def stable_sin(theta, rtol=1e-05, atol=1e-08):
    r"""Version of sin that reduces numerical error by reducing the range
    of the function to [0, 2 * pi) and handling the special cases of
    0, pi / 2, pi, 3 * pi / 2 explicitly.

    Args:
        theta (np.ndarray): Angle(s) to compute sin of.
        rtol (float, optional): Relative tolerance to use for detection
            of special cases.
        atol (float, optional): Absolute tolerance to use for detection
            of special cases.

    Returns:
        np.ndarray: Result.

    """
    rem = theta % (2.0 * np.pi)
    if isinstance(theta, np.ndarray):
        out = np.sin(rem)
        out[np.isclose(rem, 0, rtol=rtol, atol=atol)] = 0.0
        out[np.isclose(rem, np.pi / 2, rtol=rtol, atol=atol)] = 1.0
        out[np.isclose(rem, np.pi, rtol=rtol, atol=atol)] = 0.0
        out[np.isclose(rem, 3 * np.pi / 2, rtol=rtol, atol=atol)] = -1.0
        out[np.isclose(rem, 2 * np.pi, rtol=rtol, atol=atol)] = 0.0
    else:
        if ((np.isclose(rem, 0.0, rtol=rtol, atol=atol)
             or np.isclose(rem, np.pi, rtol=rtol, atol=atol)
             or np.isclose(rem, 2 * np.pi, rtol=rtol, atol=atol))):
            out = type(theta)(0.0)
        elif np.isclose(rem, np.pi / 2, rtol=rtol, atol=atol):
            out = type(theta)(1.0)
        elif np.isclose(rem, 3 * np.pi / 2, rtol=rtol, atol=atol):
            out = type(theta)(-1.0)
        else:
            out = np.sin(rem)
    return out


def stable_cos(theta, rtol=1e-05, atol=1e-08):
    r"""Version of cos that reduces numerical error by reducing the range
    of the function to [0, 2 * pi) and handling the special cases of
    0, pi / 2, pi, 3 * pi / 2 explicitly.

    Args:
        theta (np.ndarray): Angle(s) to compute cos of.
        rtol (float, optional): Relative tolerance to use for detection
            of special cases.
        atol (float, optional): Absolute tolerance to use for detection
            of special cases.

    Returns:
        np.ndarray: Result.

    """
    rem = theta % (2.0 * np.pi)
    if isinstance(theta, np.ndarray):
        out = np.cos(rem)
        out[np.isclose(rem, 0.0, rtol=rtol, atol=atol)] = 1.0
        out[np.isclose(rem, np.pi / 2, rtol=rtol, atol=atol)] = 0.0
        out[np.isclose(rem, np.pi, rtol=rtol, atol=atol)] = -1.0
        out[np.isclose(rem, 3 * np.pi / 2, rtol=rtol, atol=atol)] = 0.0
        out[np.isclose(rem, 2 * np.pi, rtol=rtol, atol=atol)] = 1.0
    else:
        if ((np.isclose(rem, np.pi / 2, rtol=rtol, atol=atol)
             or np.isclose(rem, 3 * np.pi / 2, rtol=rtol, atol=atol))):
            out = type(theta)(0.0)
        elif ((np.isclose(rem, 0, rtol=rtol, atol=atol)
               or np.isclose(rem, 2 * np.pi, rtol=rtol, atol=atol))):
            out = type(theta)(1.0)
        elif np.isclose(rem, np.pi, rtol=rtol, atol=atol):
            out = type(theta)(-1.0)
        else:
            out = np.cos(rem)
    return out


def stable_tan(theta, rtol=1e-05, atol=1e-08):
    r"""Version of tan that reduces numerical error by reducing the range
    of the function to [0, pi) and handling the special cases of
    0, pi / 4, pi / 2, 3 * pi / 4 explicitly.

    Args:
        theta (np.ndarray): Angle(s) to compute tan of.
        rtol (float, optional): Relative tolerance to use for detection
            of special cases.
        atol (float, optional): Absolute tolerance to use for detection
            of special cases.

    Returns:
        np.ndarray: Result.

    """
    rem = theta % np.pi
    if isinstance(theta, np.ndarray):
        out = np.tan(rem)
        out[np.isclose(rem, 0, rtol=rtol, atol=atol)] = 0.0
        out[np.isclose(rem, np.pi / 4, rtol=rtol, atol=atol)] = 1.0
        out[np.isclose(rem, 3 * np.pi / 4, rtol=rtol, atol=atol)] = -1.0
        out[np.isclose(rem, np.pi, rtol=rtol, atol=atol)] = 0.0
        idx_inf = np.isclose(rem, np.pi / 2, rtol=rtol, atol=atol)
        idx_neg = rem[idx_inf] > (np.pi / 2)
        out[idx_inf] = np.inf
        out[idx_inf][idx_neg] = -np.inf
    else:
        if ((np.isclose(rem, 0.0, rtol=rtol, atol=atol)
             or np.isclose(rem, np.pi, rtol=rtol, atol=atol))):
            out = type(theta)(0.0)
        elif np.isclose(rem, np.pi / 4, rtol=rtol, atol=atol):
            out = type(theta)(1.0)
        elif np.isclose(rem, np.pi / 2, rtol=rtol, atol=atol):
            out = type(theta)(np.inf)
        elif np.isclose(rem, 3 * np.pi / 4, rtol=rtol, atol=atol):
            out = type(theta)(-1.0)
        else:
            out = np.tan(rem)
    return out


def stable_arcsin(x, rtol=1e-05, atol=1e-08):
    r"""Version of arcsin that reduces numerical error by handling the
    special cases of 0, -1, and 1 explicitly.

    Args:
        x (np.ndarray): Value(s) to compute arcsin of.
        rtol (float, optional): Relative tolerance to use for detection
            of special cases.
        atol (float, optional): Absolute tolerance to use for detection
            of special cases.

    Returns:
        np.ndarray: Result.

    """
    if isinstance(x, np.ndarray):
        out = np.arcsin(x)
        out[np.isclose(x, 0, rtol=rtol, atol=atol)] = 0.0
        out[np.isclose(x, 1, rtol=rtol, atol=atol)] = np.pi / 2
        out[np.isclose(x, -1, rtol=rtol, atol=atol)] = -np.pi / 2
    elif np.isclose(x, 0, rtol=rtol, atol=atol):
        out = 0.0
    elif np.isclose(x, 1, rtol=rtol, atol=atol):
        out = np.pi / 2
    elif np.isclose(x, -1, rtol=rtol, atol=atol):
        out = -np.pi / 2
    else:
        out = np.arcsin(x)
    return out


def stable_arccos(x, rtol=1e-05, atol=1e-08):
    r"""Version of arccos that reduces numerical error by handling the
    special cases of 0, -1, and 1 explicitly.

    Args:
        x (np.ndarray): Value(s) to compute arccos of.
        rtol (float, optional): Relative tolerance to use for detection
            of special cases.
        atol (float, optional): Absolute tolerance to use for detection
            of special cases.

    Returns:
        np.ndarray: Result.

    """
    if isinstance(x, np.ndarray):
        out = np.arccos(x)
        out[np.isclose(x, 0, rtol=rtol, atol=atol)] = np.pi / 2
        out[np.isclose(x, 1, rtol=rtol, atol=atol)] = 0.0
        out[np.isclose(x, -1, rtol=rtol, atol=atol)] = np.pi
    elif np.isclose(x, 0, rtol=rtol, atol=atol):
        out = np.pi / 2
    elif np.isclose(x, 1, rtol=rtol, atol=atol):
        out = 0.0
    elif np.isclose(x, -1, rtol=rtol, atol=atol):
        out = np.pi
    else:
        out = np.arccos(x)
    return out


def stable_arctan(x, rtol=1e-05, atol=1e-08):
    r"""Version of arctan that reduces numerical error by handling the
    special cases of 0, -1, 1, and np.inf explicitly.

    Args:
        x (np.ndarray): Value(s) to compute arctan of.
        rtol (float, optional): Relative tolerance to use for detection
            of special cases.
        atol (float, optional): Absolute tolerance to use for detection
            of special cases.

    Returns:
        np.ndarray: Result.

    """
    if isinstance(x, np.ndarray):
        out = np.arctan(x)
        out[np.isclose(x, 0, rtol=rtol, atol=atol)] = 0.0
        out[np.isclose(x, 1, rtol=rtol, atol=atol)] = np.pi / 4
        out[np.isclose(x, -1, rtol=rtol, atol=atol)] = -np.pi / 4
        out[np.isclose(x, np.inf, rtol=rtol, atol=atol)] = np.pi / 2
        out[np.isclose(x, -np.inf, rtol=rtol, atol=atol)] = -np.pi / 2
    elif np.isclose(x, 0, rtol=rtol, atol=atol):
        out = 0.0
    elif np.isclose(x, 1, rtol=rtol, atol=atol):
        out = np.pi / 4
    elif np.isclose(x, -1, rtol=rtol, atol=atol):
        out = -np.pi / 4
    elif np.isclose(x, np.inf, rtol=rtol, atol=atol):
        out = np.pi / 2
    elif np.isclose(x, -np.inf, rtol=rtol, atol=atol):
        out = -np.pi / 2
    else:
        out = np.arctan(x)
    return out


def rotation_matrix(theta, u):
    r"""Get the rotation matrix necessary to rotate a 3D point around
    a unit vector by a specified angle.

    Args:
        theta (float, np.ndarray): Angle to rotate by (in radians).
            If an array is provided and u is a multidimensional array,
            theta should have the same dimensions as u outside its last
            dimension (the dimension of the unit vectors).
        u (np.ndarray): Vector to rotate around. If the array has
            multiple dimensions, the last dimension is assumed to be the
            components in each dimension and the resulting rotation
            matrix will be a composite of rotation matrices for each
            vector in u in the last two dimension such that it can be
            used to transform points with the same dimensions as u via
            numpy.matvec after swapping the last two axes.

    Returns
        np.ndarray: Rotation matrix.

    """
    norm = np.linalg.norm(u, axis=(u.ndim - 1))
    if isinstance(norm, np.ndarray):
        assert (norm > 0).all()
    else:
        assert norm > 0
    u = op_along_axis(np.divide, u, norm)
    ux = u[..., 0]
    uy = u[..., 1]
    uz = u[..., 2]
    cos_theta = stable_cos(theta)
    sin_theta = stable_sin(theta)
    inv_cos_theta = 1 - cos_theta
    dim_inner = u.ndim - 1
    dim_outer = u.ndim
    if isinstance(theta, np.ndarray) and u.ndim == 1:
        dim_inner += 1
        dim_outer += 1
    R = np.stack([
        np.stack([
            ux * ux * inv_cos_theta + cos_theta,
            ux * uy * inv_cos_theta - uz * sin_theta,
            ux * uz * inv_cos_theta + uy * sin_theta,
        ], dim_inner),
        np.stack([
            uy * ux * inv_cos_theta + uz * sin_theta,
            uy * uy * inv_cos_theta + cos_theta,
            uy * uz * inv_cos_theta - ux * sin_theta,
        ],  dim_inner),
        np.stack([
            uz * ux * inv_cos_theta - uy * sin_theta,
            uz * uy * inv_cos_theta + ux * sin_theta,
            uz * uz * inv_cos_theta + cos_theta,
        ],  dim_inner),
    ], dim_outer)
    return np.swapaxes(R, dim_inner, dim_outer)


def rotate_u(x, theta, u):
    r"""Rotate a point arount an axis by an angle.

    Args:
        x (array): Position to rotate.
        theta (float): Angle to rotate by (in radians).
        u (array): Vector to rotate around.

    Returns:
        array: Rotated position.

    """
    R = rotation_matrix(theta, u)
    if (u.ndim > 1 or isinstance(theta, np.ndarray)) and x.ndim > 1:
        if u.ndim > 1:
            assert u.shape == x.shape
        if isinstance(theta, np.ndarray):
            assert len(theta) == x.shape[0]
        if hasattr(np, 'matvec'):  # Numpy version >= 2.2.0
            return np.matvec(R, x)
        return np.vstack([  # pragma: no cover
            np.matmul(R[i, ...], x[i, ...])
            for i in range(x.shape[0])
        ])
    out = np.matmul(R, x.T)
    if x.ndim > 1:
        out = out.T
    return out


# def sun_calcs(latitude, longitude, standard_meridian, day_of_year,
#               hour_of_day):
#     solar_noon = 12

#     # LOCAL MERIDIAN IN DEGREES
#     local_meridian = standard_meridian - longitude
#     # print("local meridian")
#     # print(local_meridian)

#     # radians_per_degree = np.pi / 180

#     # radians_rotation_per_hour = 15 * radians_per_degree
#     # axial_tilt = 13.5 * radians_per_degree

#     # phi = latitude * radians_per_degree

#     # from sun model code in HackMD: https://hackmd.io/98rj0V82RSyrDavaFxQbEg
#     # number_days_after_dec_solstice = day_of_year + 10

#     # omega = (
#     #     360.0 * (number_days_after_dec_solstice / 365.0)
#     #     * radians_per_degree
#     # )
#     # delta = -axial_tilt * math.cos(omega)
#     # tau = (hour_of_day - solar_noon) * radians_rotation_per_hour

#     # equation of time (in hours)
#     J = day_of_year
#     f = 279.575 + (0.9856 * J)

#     f_rad = math.radians(f)
#     part_1 = math.sin(f_rad) * -104.7

#     f2 = f * 2
#     f2_rad = math.radians(f2)
#     part_2 = math.sin(f2_rad) * 596.2

#     f3 = f * 3
#     f3_rad = math.radians(f3)
#     part_3 = math.sin(f3_rad) * 4.3

#     f4 = f * 4
#     f4_rad = math.radians(f4)
#     part_4 = math.sin(f4_rad) * 12.7

#     part_5 = math.cos(f_rad) * 429.3

#     part_6 = math.cos(f2_rad) * 2.0

#     part_7 = math.cos(f3_rad) * 19.3

#     all_together = (
#         part_1 + part_2 + part_3 - part_4 - part_5 - part_6 + part_7
#     )

#     equation_of_time = all_together / 3600
#     # print("equation of time")
#     # print(equation_of_time)

#     longitudinal_correction = local_meridian / 15  # in hours
#     # print("logitudinal correction")
#     # print(longitudinal_correction)

#     # time of solar noon (in hours)
#     t0 = 12 - longitudinal_correction - equation_of_time
#     # print("t0")
#     # print(t0)

#     # solar declination
#     # should range from +23.45 degrees at summer solstice to -23.45
#     #     degrees at winter solstice
#     times_j = 0.9856 * J
#     inside_sine = 356.6 + times_j
#     inside_sine_rad = math.radians(inside_sine)

#     inner_sin = math.sin(inside_sine_rad)

#     additions = 278.97 + times_j + (1.9165 * inner_sin)
#     additions_rad = math.radians(additions)
#     sin_of_adds = math.sin(additions_rad)
#     times_const = 0.39785 * sin_of_adds

#     final_in_rad = math.asin(times_const)

#     solar_declination = math.degrees(final_in_rad)
#     # print("solar declination")
#     # print(solar_declination)

#     # calculate zenith angle

#     latitude_rad = math.radians(latitude)
#     solar_declination_rad = math.radians(solar_declination)

#     sin_1 = math.sin(latitude_rad)
#     sin_2 = math.sin(solar_declination_rad)

#     cos_1 = math.cos(latitude_rad)
#     cos_2 = math.cos(solar_declination_rad)

#     inside_cos = 15 * (hour_of_day - t0)
#     inside_cos_rad = math.radians(inside_cos)

#     cos_3 = math.cos(inside_cos_rad)

#     before_arccos = (sin_1 * sin_2) + (cos_1 * cos_2 * cos_3)

#     zenith_angle_rad = math.acos(before_arccos)

#     zenith_angle = math.degrees(zenith_angle_rad)
#     # print("zenith angle")
#     # print(zenith_angle)

#     ninety_six_rad = math.radians(96)
#     first_cos = math.cos(ninety_six_rad)

#     sins = math.sin(latitude_rad) * math.sin(solar_declination_rad)

#     cosines = math.cos(latitude_rad) * math.cos(solar_declination_rad)

#     equation = (first_cos - sins) / cosines

#     # FIX: get error at certain latitudes/longitudes
#     equation_acos = math.acos(equation)

#     half_daylength = math.degrees(equation_acos)
#     # print("half daylength")
#     # print(half_daylength)

#     time_sunrise = solar_noon - (half_daylength / 15)
#     # print("time sunrise")
#     # print(time_sunrise)

#     hours = half_daylength * (1 / 15)
#     # print("hours")
#     # print(hours)

#     first_twilight = t0 - hours
#     # print("first twilight")
#     # print(first_twilight)

#     daylength = 2 * hours
#     # print("daylength")
#     # print(daylength)

#     # FIX: including a +1 to take care of daylight savings time for now
#     time_sunrise = (
#         first_twilight + longitudinal_correction + equation_of_time + 1
#     )
#     # print("time sunrise")
#     # print(time_sunrise)

#     # return zenith_angle
#     return {
#         'zenith': zenith_angle, 'half_daylength': half_daylength,
#         'time_sunrise': time_sunrise, 'hours': hours,
#         'first_twilight': first_twilight, 'daylength': daylength
#     }


# #######
# def sun_energy_total():
#     # surface temp of sun
#     T = 5800  # Kelvin

#     # radius of sun
#     R = 6.995 * 10**8  # meters

#     # stefan's constant
#     stefan = 5.67 * 10**-8  # Wm^-2K^-4

#     # equation for thermal energy per second (power) radiated by sun
#     power = 4 * np.pi * (R)**2 * stefan * (T)**4  # Watts
#     print("total sun energy per second:", power)

#     # per minute
#     energy = power * 60  # Joules
#     print("total sun energy per minute:", energy)


# ########
# def power_atmosphere(zenith_angle, day_of_year):

#     # sun_surface_energy = 6.33 * 10**7  # watts / meter squared

#     # equals (6.33 * 10**7 W/m^2)*(surface area of sun)
#     #    / (4pi)((distance form earth to sun)^2)
#     solar_constant = 1367  # watts / meter squared

#     # outside of the atmosphere
#     extraterrestrial_irradiance = (
#         solar_constant * (1 + 0.034 * math.cos((2 * np.pi)
#                                                * (day_of_year / 265.25)))
#     )
#     # print(extraterrestrial_irradiance)

#     cos_zenith_angle = math.cos(zenith_angle)
#     # print(cos_zenith_angle)

#     # instantaneous power at this time
#     power_without_atmosphere = (  # watts / meter squared
#         extraterrestrial_irradiance * cos_zenith_angle
#     )
#     # print(power_without_atmosphere)

#     # atmospheric effects
#     # take into account cloudy vs. sunny
#     # rayleigh scattering

#     # equation 2.3: how much air the sun has to go through to get to a
#     #     point (not taking into account weather/clouds)
#     # ranges from 0 to 2 --> amount getting blocked by the atmosphere

#     # constants
#     radius_of_earth = 6370  # kilometers
#     # FIX: need to figure out how to adjust based on location on earth
#     thickness_of_atmosphere = 7991  # kilometers
#     # atmospheric effects --> does not take into account various
#     #     atmospheric effects
#     air_mass = (
#         (((radius_of_earth / thickness_of_atmosphere)
#           * math.cos(zenith_angle))**2
#          + (2 * (radius_of_earth / thickness_of_atmosphere) + 1)) ** 0.5
#         - ((radius_of_earth / thickness_of_atmosphere)
#            * math.cos(zenith_angle))
#     )
#     # print(air_mass)

#     half_air_mass = air_mass / 2
#     # print(half_air_mass)

#     power_with_atmosphere = (  # watts / meter squared
#         power_without_atmosphere * half_air_mass
#     )
#     print("power_with_atmosphere: ", power_with_atmosphere)

#     return power_with_atmosphere


# ########
# def photons_output(power_with_atmosphere):
#     plancks_constant = 6.626 * 10**-34  # Js

#     sample_wavelength = 635  # nanometers (nm)
#     sample_wavelength_meter = sample_wavelength * (1 * 10**-9)
#     C = 3.0 * 10**8  # m/s
#     frequency = C / sample_wavelength_meter  # vibrations per second
#     print(frequency)

#     # single photon = h * C / lamba (in Joules)
#     single_photon = plancks_constant * C / frequency  # Joules
#     print(single_photon)

#     # 1 J/(s * m^2) = watt / m^2
#     energy_photon = plancks_constant * frequency  # Joules
#     print(energy_photon)

#     # calculate number of photons?
