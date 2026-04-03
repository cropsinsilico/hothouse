import pytest
import numpy as np
from hothouse import sun_calc


def test_op_along_axis(assert_allclose):
    r"""Test elementwise operation along axis."""
    a = np.ones((5, 3))
    b = np.arange(1, 6)
    assert_allclose(
        sun_calc.op_along_axis(np.multiply, a, b),
        np.vstack([b, b, b]).T
    )
    assert_allclose(
        sun_calc.op_along_axis(np.divide, a, b),
        np.vstack([1 / b, 1 / b, 1 / b]).T
    )


def test_stable_sin(assert_allclose):
    r"""Test stable_sin."""
    x = np.linspace(0, 2 * np.pi, 100)
    assert_allclose(sun_calc.stable_sin(x), np.sin(x))
    x = (np.pi / 2) * np.arange(4)
    y = np.array([0.0, 1.0, 0.0, -1.0])
    for xx, yy in zip(x, y):
        assert_allclose(sun_calc.stable_sin(xx), yy)
    nrep = 5
    x = np.hstack([x + i * 2 * np.pi for i in range(nrep)]
                  + [x - i * 2 * np.pi for i in range(nrep)])
    y = np.hstack(2 * nrep * [y])
    assert_allclose(sun_calc.stable_sin(x), y)


def test_stable_cos(assert_allclose):
    r"""Test stable_cos."""
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    assert_allclose(sun_calc.stable_cos(x), np.cos(x))
    x = (np.pi / 2) * np.arange(4)
    y = np.array([1.0, 0.0, -1.0, 0.0])
    for xx, yy in zip(x, y):
        assert_allclose(sun_calc.stable_cos(xx), yy)
    nrep = 5
    x = np.hstack([x + i * 2 * np.pi for i in range(nrep)]
                  + [x - i * 2 * np.pi for i in range(nrep)])
    y = np.hstack(2 * nrep * [y])
    assert_allclose(sun_calc.stable_cos(x), y)


def test_stable_tan(assert_allclose):
    r"""Test stable_tan."""
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    assert_allclose(sun_calc.stable_tan(x), np.tan(x))
    x = (np.pi / 4) * np.arange(4)
    y = np.array([0.0, 1.0, np.inf, -1.0])
    for xx, yy in zip(x, y):
        assert_allclose(sun_calc.stable_tan(xx), yy)
    nrep = 5
    x = np.hstack([x + i * np.pi for i in range(nrep)]
                  + [x - i * np.pi for i in range(nrep)])
    y = np.hstack(2 * nrep * [y])
    assert_allclose(sun_calc.stable_tan(x), y)


def test_stable_arcsin(assert_allclose):
    r"""Test stable_arcsin."""
    x = np.linspace(-1, 1, 100)
    assert_allclose(sun_calc.stable_arcsin(x), np.arcsin(x))
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-np.pi / 2, 0.0, np.pi / 2])
    for xx, yy in zip(x, y):
        assert_allclose(sun_calc.stable_arcsin(xx), yy)
    nrep = 5
    atol = 1e-09
    x = np.hstack([x + i * atol for i in range(nrep)]
                  + [x - i * atol for i in range(nrep)])
    x[x < -1] = -1
    x[x > 1] = 1
    y = np.hstack(2 * nrep * [y])
    assert_allclose(sun_calc.stable_arcsin(x), y)


def test_stable_arccos(assert_allclose):
    r"""Test stable_arccos."""
    x = np.linspace(-1, 1, 100)
    assert_allclose(sun_calc.stable_arccos(x), np.arccos(x))
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([np.pi, np.pi / 2, 0.0])
    for xx, yy in zip(x, y):
        assert_allclose(sun_calc.stable_arccos(xx), yy)
    nrep = 5
    atol = 1e-9
    x = np.hstack([x + i * atol for i in range(nrep)]
                  + [x - i * atol for i in range(nrep)])
    x[x < -1] = -1
    x[x > 1] = 1
    y = np.hstack(2 * nrep * [y])
    assert_allclose(sun_calc.stable_arccos(x), y)


def test_stable_arctan(assert_allclose):
    r"""Test stable_arctan."""
    x = np.linspace(-1, 1, 100)
    assert_allclose(sun_calc.stable_arctan(x), np.arctan(x))
    x = np.array([-1.0, 0.0, 1.0, np.inf, -np.inf])
    y = np.array([-np.pi / 4, 0.0, np.pi / 4, np.pi / 2, -np.pi / 2])
    for xx, yy in zip(x, y):
        assert_allclose(sun_calc.stable_arctan(xx), yy)
    nrep = 5
    atol = 1e-09
    x = np.hstack([x + i * atol for i in range(nrep)]
                  + [x - i * atol for i in range(nrep)])
    y = np.hstack(2 * nrep * [y])
    assert_allclose(sun_calc.stable_arctan(x), y)


def test_rotate_u_x(uvec_x, uvec_y, uvec_z, assert_allclose):
    r"""Test rotation equation around x."""
    u = uvec_x
    assert_allclose(sun_calc.rotate_u(uvec_x, np.pi/2.0, u), uvec_x)
    assert_allclose(sun_calc.rotate_u(uvec_y, np.pi/2.0, u), uvec_z)
    assert_allclose(sun_calc.rotate_u(uvec_z, np.pi/2.0, u), -uvec_y)
    xmat = np.vstack([uvec_x, uvec_y, uvec_z, uvec_z])
    xres = np.vstack([uvec_x, uvec_z, -uvec_y, -uvec_y])
    assert_allclose(sun_calc.rotate_u(xmat, np.pi/2.0, u), xres)


def test_rotate_u_y(uvec_x, uvec_y, uvec_z, assert_allclose):
    r"""Test rotation equation around y."""
    u = uvec_y
    assert_allclose(sun_calc.rotate_u(uvec_x, np.pi/2.0, u), -uvec_z)
    assert_allclose(sun_calc.rotate_u(uvec_y, np.pi/2.0, u), uvec_y)
    assert_allclose(sun_calc.rotate_u(uvec_z, np.pi/2.0, u), uvec_x)
    xmat = np.vstack([uvec_x, uvec_y, uvec_z, uvec_z])
    xres = np.vstack([-uvec_z, uvec_y, uvec_x, uvec_x])
    assert_allclose(sun_calc.rotate_u(xmat, np.pi/2.0, u), xres)


def test_rotate_u_z(uvec_x, uvec_y, uvec_z, assert_allclose):
    r"""Test rotation equation around z."""
    u = uvec_z
    assert_allclose(sun_calc.rotate_u(uvec_x, np.pi/2.0, u), uvec_y)
    assert_allclose(sun_calc.rotate_u(uvec_y, np.pi/2.0, u), -uvec_x)
    assert_allclose(sun_calc.rotate_u(uvec_z, np.pi/2.0, u), uvec_z)
    xmat = np.vstack([uvec_x, uvec_y, uvec_z, uvec_y])
    xres = np.vstack([uvec_y, -uvec_x, uvec_z, -uvec_x])
    assert_allclose(sun_calc.rotate_u(xmat, np.pi/2.0, u), xres)


def test_rotate_umat(uvec_x, uvec_y, uvec_z, assert_allclose):
    r"""Test rotation around array of vectors."""
    u = np.vstack([uvec_x, uvec_y, uvec_z, uvec_z])
    x = np.vstack([uvec_y, uvec_z, uvec_x, uvec_x])
    y = np.vstack([uvec_z, uvec_x, uvec_y, uvec_y])
    for i in range(x.shape[0]):
        ires = sun_calc.rotate_u(x[i, :], np.pi/2.0, u)
        assert ires.shape == u.shape
        for j in range(u.shape[0]):
            jexp = sun_calc.rotate_u(x[i, :], np.pi/2.0, u[j, :])
            assert_allclose(ires[j, :], jexp)
    assert_allclose(sun_calc.rotate_u(x, np.pi/2.0, u), y)


def test_rotate_theta_mat(uvec_x, uvec_y, uvec_z, assert_allclose):
    r"""Test rotation by array of angles."""
    theta = np.linspace(0, 2 * np.pi, 5)
    assert_allclose(sun_calc.rotate_u(uvec_y, theta, uvec_x),
                    np.vstack([uvec_y, uvec_z, -uvec_y, -uvec_z, uvec_y]))
    u = np.vstack([uvec_x, uvec_y, uvec_z, uvec_z, uvec_x])
    assert_allclose(sun_calc.rotate_u(uvec_y, theta, u),
                    np.vstack([uvec_y, uvec_y, -uvec_y,
                               uvec_x, uvec_y]))
    assert_allclose(sun_calc.rotate_u(np.vstack(5 * [uvec_y]), theta, u),
                    np.vstack([uvec_y, uvec_y, -uvec_y,
                               uvec_x, uvec_y]))
    assert_allclose(sun_calc.rotate_u(np.vstack(5 * [uvec_y]),
                                      theta, uvec_x),
                    np.vstack([uvec_y, uvec_z, -uvec_y, -uvec_z,
                               uvec_y]))


@pytest.mark.parametrize("date_name,direct,diffuse", [
    ("noon", 1366.05821092, 253.30850887),
    ("sunrise", 245.03424107, 46.11208444),
    ("sunset", 0.9476334, 0.35103056),
])
def test_ppfd(date_name, direct, diffuse, location_champaign,
              altitude_champaign, datetime_champaign, assert_allclose):
    r"""Test PPFD calculation."""
    date = datetime_champaign(date_name)
    ppfd_tot = sun_calc.solar_ppfd(*location_champaign, date,
                                   altitude=altitude_champaign)
    assert_allclose(ppfd_tot['direct'], direct)
    assert_allclose(ppfd_tot['diffuse'], diffuse)
