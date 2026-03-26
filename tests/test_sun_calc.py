import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from hothouse import sun_calc


def test_op_along_axis():
    r"""Test elementwise operation along axis."""
    a = np.ones((5, 3))
    b = np.arange(5)
    assert_almost_equal(
        sun_calc.op_along_axis(np.multiply, a, b),
        np.vstack([b, b, b]).T
    )
    assert_almost_equal(
        sun_calc.op_along_axis(np.divide, a, b),
        np.vstack([1 / b, 1 / b, 1 / b]).T
    )


def test_rotate_u_x(uvec_x, uvec_y, uvec_z):
    r"""Test rotation equation around x."""
    u = uvec_x
    assert_almost_equal(sun_calc.rotate_u(uvec_x, np.pi/2.0, u), uvec_x)
    assert_almost_equal(sun_calc.rotate_u(uvec_y, np.pi/2.0, u), uvec_z)
    assert_almost_equal(sun_calc.rotate_u(uvec_z, np.pi/2.0, u), -uvec_y)
    xmat = np.vstack([uvec_x, uvec_y, uvec_z, uvec_z])
    xres = np.vstack([uvec_x, uvec_z, -uvec_y, -uvec_y])
    assert_almost_equal(sun_calc.rotate_u(xmat, np.pi/2.0, u), xres)


def test_rotate_u_y(uvec_x, uvec_y, uvec_z):
    r"""Test rotation equation around y."""
    u = uvec_y
    assert_almost_equal(sun_calc.rotate_u(uvec_x, np.pi/2.0, u), -uvec_z)
    assert_almost_equal(sun_calc.rotate_u(uvec_y, np.pi/2.0, u), uvec_y)
    assert_almost_equal(sun_calc.rotate_u(uvec_z, np.pi/2.0, u), uvec_x)
    xmat = np.vstack([uvec_x, uvec_y, uvec_z, uvec_z])
    xres = np.vstack([-uvec_z, uvec_y, uvec_x, uvec_x])
    assert_almost_equal(sun_calc.rotate_u(xmat, np.pi/2.0, u), xres)


def test_rotate_u_z(uvec_x, uvec_y, uvec_z):
    r"""Test rotation equation around z."""
    u = uvec_z
    assert_almost_equal(sun_calc.rotate_u(uvec_x, np.pi/2.0, u), uvec_y)
    assert_almost_equal(sun_calc.rotate_u(uvec_y, np.pi/2.0, u), -uvec_x)
    assert_almost_equal(sun_calc.rotate_u(uvec_z, np.pi/2.0, u), uvec_z)
    xmat = np.vstack([uvec_x, uvec_y, uvec_z, uvec_y])
    xres = np.vstack([uvec_y, -uvec_x, uvec_z, -uvec_x])
    assert_almost_equal(sun_calc.rotate_u(xmat, np.pi/2.0, u), xres)


def test_rotate_umat(uvec_x, uvec_y, uvec_z):
    r"""Test rotation around array of vectors."""
    u = np.vstack([uvec_x, uvec_y, uvec_z, uvec_z])
    x = np.vstack([uvec_y, uvec_z, uvec_x, uvec_x])
    y = np.vstack([uvec_z, uvec_x, uvec_y, uvec_y])
    for i in range(x.shape[0]):
        ires = sun_calc.rotate_u(x[i, :], np.pi/2.0, u)
        assert ires.shape == u.shape
        for j in range(u.shape[0]):
            jexp = sun_calc.rotate_u(x[i, :], np.pi/2.0, u[j, :])
            assert_almost_equal(ires[j, :], jexp)
    assert_almost_equal(sun_calc.rotate_u(x, np.pi/2.0, u), y)


def test_rotate_theta_mat(uvec_x, uvec_y, uvec_z):
    r"""Test rotation by array of angles."""
    theta = np.linspace(0, 2 * np.pi, 5)
    assert_almost_equal(sun_calc.rotate_u(uvec_y, theta, uvec_x),
                        np.vstack([uvec_y, uvec_z, -uvec_y, -uvec_z,
                                   uvec_y]))
    u = np.vstack([uvec_x, uvec_y, uvec_z, uvec_z, uvec_x])
    assert_almost_equal(sun_calc.rotate_u(uvec_y, theta, u),
                        np.vstack([uvec_y, uvec_y, -uvec_y,
                                   uvec_x, uvec_y]))
    assert_almost_equal(sun_calc.rotate_u(np.vstack(5 * [uvec_y]), theta, u),
                        np.vstack([uvec_y, uvec_y, -uvec_y,
                                   uvec_x, uvec_y]))
    assert_almost_equal(sun_calc.rotate_u(np.vstack(5 * [uvec_y]),
                                          theta, uvec_x),
                        np.vstack([uvec_y, uvec_z, -uvec_y, -uvec_z,
                                   uvec_y]))


@pytest.mark.parametrize("date_name,direct,diffuse", [
    ("noon", 1366.05821092, 253.30850887),
    ("sunrise", 245.03424107, 46.11208444),
    ("sunset", 0.9476334, 0.35103056),
])
def test_ppfd(date_name, direct, diffuse, location_champaign,
              altitude_champaign, datetime_champaign):
    r"""Test PPFD calculation."""
    date = datetime_champaign(date_name)
    ppfd_tot = sun_calc.solar_ppfd(*location_champaign, date,
                                   altitude=altitude_champaign)
    assert_almost_equal(ppfd_tot['direct'], direct)
    assert_almost_equal(ppfd_tot['diffuse'], diffuse)
