from datetime import datetime, timezone

import numpy as np
import pytest

from mwir_control.geometry import Camera, TriangleScene, heading_basis, observer_offset, ray_box_interval
from mwir_control.resources import PublicResources, read_element_mapping, sample_spectrum


def test_camera_and_mesh_quadrants():
    camera = Camera((3, 3), 2, 2, 1, 1, [0, 0, 0], np.eye(3))
    rays = camera.rays()
    assert np.allclose(rays[1, 1], [0, 0, 1])
    assert np.allclose(camera.project(rays * 7)[0, 2], [2, 0])
    frame = PublicResources().ship_frame()
    body = heading_basis(0)
    assert np.allclose(frame.mesh_to_enu(body, [0, 0, 0], np.eye(3)), [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    fwd, starboard = body[:, :2].T
    assert observer_offset(fwd, starboard, "lf") @ starboard < 0
    assert observer_offset(fwd, starboard, "rf") @ starboard > 0
    assert observer_offset(fwd, starboard, "lr") @ fwd < 0


def test_triangle_depth_and_box_occlusion():
    vertices = np.array([[-1, -1, 4], [1, -1, 4], [0, 1, 4]])
    scene = TriangleScene(np.vstack([vertices, vertices + [0, 0, 2]]), [[0, 1, 2], [3, 4, 5]])
    rays = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1]])
    hits = scene.intersect([0, 0, 0], rays)
    np.testing.assert_array_equal(hits.triangle_index, [0, -1, -1])
    np.testing.assert_allclose(hits.distance_m, [4, np.inf, np.inf])
    np.testing.assert_allclose(hits.normal[0], [0, 0, -1])
    entry, exit = ray_box_interval([0, 0, 0], rays, [-1, -1, 3], [1, 1, 5])
    np.testing.assert_allclose(entry, [3, np.inf, np.inf])
    np.testing.assert_allclose(exit, [5, np.inf, np.inf])
    assert min(exit[0], hits.distance_m[0]) - entry[0] == 1
    inside = ray_box_interval([0, 0, 4], [1, 0, 0], [-1, -1, 3], [1, 1, 5])
    assert inside == (0, 1)


def test_pinned_resources_and_units():
    public = PublicResources()
    public.verify()
    ship, sea = public.mesh("ship_mesh"), public.mesh("sea_mesh")
    assert len(ship.triangles) == 18533 and len(sea.triangles) == 80600
    assert np.max(np.abs(ship.vertices_m)) < 100
    mapping = read_element_mapping(public.path("element_mapping"))
    assert all(mapping[eid][0] == pid for eid, pid in zip(ship.element_id, ship.property_id) if eid in mapping)
    assert public.stack_outlets()["short_stack"][0][2] == pytest.approx(14.4257)
    rsr = public.broadband_response()
    assert rsr.shape == (93, 2) and np.all(np.diff(rsr[:, 0]) > 0)
    water = public.water_nk()
    assert sample_spectrum(water, [3, 5], "water").shape == (2, 2)
    with pytest.raises(ValueError, match="coverage"):
        sample_spectrum(water, [2.99], "water")


def test_table_time_and_geometry_are_explicit():
    public = PublicResources()
    date = datetime(1984, 7, 19, 9, 54, tzinfo=timezone.utc)
    assert public.solar_at(date).shape == (251, 2)
    table = public.atmosphere_at(date, height_m=150, range_m=440, elevation_deg=20)
    assert table.shape == (1334, 4)
    assert np.all((table[:, 1] >= 0) & (table[:, 1] <= 1))
    with pytest.raises(ValueError, match="height/range/elevation"):
        public.atmosphere_at(date, height_m=150, range_m=438, elevation_deg=20)
    with pytest.raises(ValueError, match="no exact"):
        public.solar_at(date.replace(minute=55))
