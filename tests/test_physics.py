import numpy as np
import pytest

from mwir_control.nozzles import Action, Nozzle, NozzleRegistry
from mwir_control.physics import (Atmosphere, Boundary, NozzleCalibration, Plume, RTPCM,
                                  RegionCalibration, SpectralGrid, SpraySystem)
from mwir_control.physics_mist import (Curtain, droplet_number_density, effective_radius_m,
                                       liquid_water_content, mie_optics, orientation_weight)
from mwir_control.physics_spectral import (classify_los, kruse_atmosphere, planck_um,
                                           plume_radiance, sea_directional_reflectance, sea_radiance)


def spectral_grid():
    wavelengths = [3, 3.5, 3.85, 3.9, 3.95, 4.2, 5]
    response = [[1, 1, 1, 1, 1, 1, 1], [0, 0, .5, 1, .5, 0, 0]]
    return SpectralGrid(wavelengths, response, [[3, 5], [3.85, 3.95]], [.1, .2])


def test_planck_and_kruse_units():
    wavelength = np.array([3.0, 4.0, 5.0])
    assert planck_um(wavelength, 300)[1] == pytest.approx(0.7219764)
    assert np.all(planck_um(wavelength, 360) > planck_um(wavelength, 300))
    atm = kruse_atmosphere(wavelength, 10, 438, 300)
    depth = (3.912 / 10) * (4 / .55)**(-1.3) * .438
    assert atm.transmittance[1] == pytest.approx(np.exp(-depth))
    np.testing.assert_allclose(atm.path_radiance, (1 - atm.transmittance) * planck_um(wavelength, 300))
    transparent = kruse_atmosphere(wavelength, 10, 0, 300)
    np.testing.assert_equal(transparent.transmittance, np.ones(3))
    np.testing.assert_equal(transparent.path_radiance, np.zeros(3))
    with pytest.raises(ValueError, match="visibility_km"):
        kruse_atmosphere(wavelength, 1, 438, 300)


def test_band_integral_and_background_inverse():
    grid = spectral_grid()
    constant = np.full(7, 8.0)
    assert grid.integrate(constant, 0) == pytest.approx(16)
    assert grid.integrate(constant, 1) == pytest.approx(8 * .075)
    np.testing.assert_allclose(grid.equivalent_solar(constant), (8, 8))
    means = np.array([1.2, .1])
    background = grid.background(means)
    np.testing.assert_allclose([grid.integrate(background, c) for c in range(2)], means)
    with pytest.raises(ValueError, match="multiple"):
        SpectralGrid(grid.wavelength_um, [[1]*7, [0, 0, 0, 1, 0, 0, 0]], grid.bands_um, grid.noise)


def test_plume_order_surface_tie_and_sea_source():
    surface, plume, background = classify_los([1, 3, np.inf], [1, 1, np.inf])
    np.testing.assert_equal(surface, [True, False, False])
    np.testing.assert_equal(plume, [False, True, False])
    np.testing.assert_equal(background, [False, False, True])
    wavelength = np.array([4.0])
    temperatures = [700, 500]
    radiance = plume_radiance(wavelength, temperatures, [[1.0], [2.0]], [0.5, 1.0])
    emission = planck_um(wavelength, temperatures)[:, 0]
    reference = emission[0] * (1 - np.exp(-.5)) * np.exp(-2) + emission[1] * (1 - np.exp(-2))
    assert radiance[0] == pytest.approx(reference)
    assert sea_radiance(wavelength, 300, 0, .2, .5, 10)[0] == pytest.approx(1 / np.pi)


def test_curtain_chord_tangency_and_effective_area():
    calibration = RegionCalibration(2, 1, 0, 10e-6, 1)
    curtain = Curtain(np.array([[0, 0, 0], [0, 0, 2], [0, 0, 0]]),
                      np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0]), calibration)
    assert calibration.area_m2() == pytest.approx(np.pi)
    entry, exit = curtain.intersect(np.array([-2, 0, 1]), np.array([[1., 0, 0], [0, 0, 1]]))
    np.testing.assert_allclose(entry, [1, np.inf])
    np.testing.assert_allclose(exit, [3, np.inf])
    entry, exit = curtain.intersect(np.array([0, 0, -2]), np.array([[0., 0, 1]]))
    np.testing.assert_allclose([entry[0], exit[0]], [2, 4])
    calibration = RegionCalibration(2, .1, 0, 10e-6, 1, 2, 1)
    assert calibration.area_m2() == pytest.approx(.075 * np.pi)


def test_curved_curtain_and_angular_sea_calibration():
    # At s/length=1/2, the centre of this transverse section is x=0.25.
    calibration = RegionCalibration(2, .1, 0, 10e-6, 1)
    curtain = Curtain(np.array([[0, 0, 0], [0, 0, 2], [1, 0, 0]]),
                      np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0]), calibration)
    entry, exit = curtain.intersect(np.array([.25, -1, 1]), np.array([[0, 1., 0]]))
    np.testing.assert_allclose([entry[0], exit[0]], [.9, 1.1])
    direction = np.sqrt(.5) * np.array([1, 0, 1])
    factor = sea_directional_reflectance(-direction[None], direction, [0, 0, 1],
                                        [0, 90], [0, 90], [[.1, .2], [.3, .4]])
    assert factor[0] == pytest.approx(.25)


def test_mie_mass_balance_and_printed_orientation_kernel():
    x = .01
    extinction, albedo, asymmetry = mie_optics([4.0], [1.33 + 0j], x * 4e-6 / (2 * np.pi))
    factor = (1.33**2 - 1) / (1.33**2 + 2)
    assert extinction[0] == pytest.approx(8 / 3 * x**4 * factor**2, rel=1e-3)
    assert albedo[0] == pytest.approx(1)
    assert abs(asymmetry[0]) < 1e-3
    extinction, albedo, _ = mie_optics([4.0], [1.33 + .08j], 3e-6)
    assert extinction[0] > 0
    assert 0 < albedo[0] < 1
    assert orientation_weight(0.0, 1.0) == pytest.approx((4 * np.pi)**-1.5)
    assert orientation_weight(.9, 1.0) == pytest.approx((1.9 / np.sqrt(4 * np.pi))**3)
    radius = effective_radius_m(20e-6, 1.4)
    assert radius == pytest.approx(10e-6 * np.exp(2.5 * np.log(1.4)**2))
    lwc = liquid_water_content(60, 2, 1, 1000)
    assert lwc == pytest.approx(.5)
    density = droplet_number_density(lwc, radius, 1000)
    assert density * 1000 * 4 / 3 * np.pi * radius**3 == pytest.approx(lwc)


def test_forward_recalculates_mist_with_candidate_and_foreground():
    grid = spectral_grid()
    registry = NozzleRegistry([Nozzle("n", "SC_1", "SC", [0, 0, 0], np.eye(3), [0, 1, 0], .207)])
    region = RegionCalibration(2, .1, 0, 20e-6, 1.2, .4, 1)
    calibration = NozzleCalibration([.2, .36], [12, 15], [[0, 0, 0], [0, 0, 0]], [2, 2])
    spray = SpraySystem(np.eye(3), [0, 0, 0], [0, 0, 0], np.full(7, 1.33 + .08j), 1000,
                        {"SC_1": region}, {"n": calibration})
    rays = np.array([[[0., 0, 1], [0, 0, 1], [0, 1, 0]]])
    atmosphere = Atmosphere(np.full(7, .8), np.full(7, .25), "external:test")
    boundary = Boundary([0, 0, -2], rays, [[5, 1, np.inf]], [[600, 600, np.nan]], 1,
                        [[False, False, True]], [1, .05], 0, .1, .99, 280, np.zeros(7), atmosphere)
    model = RTPCM(grid, boundary, registry, spray)
    no_mist = model.evaluate(Action.off(1))
    first = model.evaluate(Action([[.23, .4, 0]], [True]))
    second = model.evaluate(Action([[.32, .9, 0]], [True]))
    steer = model.evaluate(Action([[.23, .4, 8]], [True]))
    assert no_mist.atmosphere_source == "external:test"
    np.testing.assert_equal(no_mist.fog_chord_m, [[0, 0, 0]])
    np.testing.assert_equal(first.fog_region, [[0, -1, -1]])
    np.testing.assert_allclose(first.fog_chord_m, [[2, 0, 0]])
    np.testing.assert_allclose(no_mist.frames[:, 0, 1:], first.frames[:, 0, 1:])
    assert np.all(second.frames[:, 0, 0] < first.frames[:, 0, 0])
    assert steer.fog_chord_m[0, 0] < first.fog_chord_m[0, 0]
    assert no_mist.frames[0, 0, 0] == pytest.approx(grid.integrate(.8 * planck_um(grid.wavelength_um, 600) + .25, 0))

