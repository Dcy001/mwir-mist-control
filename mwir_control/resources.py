"""Explicit readers for the pinned viewer resources and calibrated spectra."""

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .geometry import heading_basis, orthogonal, vec3


PUBLIC_ROOT = Path(__file__).with_name("public")
SOURCE_MANIFEST = PUBLIC_ROOT / "manifest.json"


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def spectral_columns(path, *columns):
    """Read explicit numeric CSV columns with strictly increasing wavelength."""
    try:
        values = np.array([[float(row[field]) for field in columns] for row in _rows(path)])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"{path}: expected numeric columns {columns}") from exc
    return check_spectrum(values, str(path))


def check_spectrum(values, field):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all()
            or (values[:, 0] <= 0).any() or (np.diff(values[:, 0]) <= 0).any()):
        raise ValueError(f"{field}: need finite values and increasing positive wavelengths in µm")
    return values


def sample_spectrum(values, wavelength_um, field):
    """Interpolate within the calibrated wavelength coverage only."""
    values = check_spectrum(values, field)
    grid = np.asarray(wavelength_um, dtype=float)
    if not np.isfinite(grid).all() or np.any((grid < values[0, 0]) | (grid > values[-1, 0])):
        raise ValueError(f"{field}: requested wavelength outside supplied coverage")
    return np.stack([np.interp(grid, values[:, 0], column) for column in values[:, 1:].T], axis=-1)


@dataclass(frozen=True)
class Mesh:
    vertices_m: np.ndarray
    triangles: np.ndarray
    element_id: np.ndarray
    property_id: np.ndarray


def read_nastran(path, *, unit):
    """Free-field GRID, CTRIA3 and CQUAD4 in the basic coordinate system."""
    if unit not in {"mm", "m"}:
        raise ValueError("Mesh unit must be explicitly 'mm' or 'm'")
    nodes, elements = {}, {}
    with Path(path).open(encoding="latin1") as stream:
        for line_number, line in enumerate(stream, 1):
            record = line.partition("$")[0].strip()
            fields = [field.strip() for field in record.split(",")]
            card = fields[0]
            try:
                if card == "GRID":
                    key = int(fields[1])
                    if key in nodes or fields[2] not in {"", "0"}:
                        raise ValueError("duplicate GRID or non-basic coordinate system")
                    nodes[key] = tuple(float(part.replace("D", "E")) for part in fields[3:6])
                    if len(nodes[key]) != 3:
                        raise ValueError("GRID needs three coordinates")
                elif card in {"CTRIA3", "CQUAD4"}:
                    key, pid = int(fields[1]), int(fields[2])
                    count = 3 if card == "CTRIA3" else 4
                    ids = tuple(int(part) for part in fields[3:3 + count])
                    if key in elements or len(ids) != count:
                        raise ValueError("duplicate or incomplete element")
                    elements[key] = pid, ids
                elif card.split() and card.split()[0].rstrip("*") in {"GRID", "CTRIA3", "CQUAD4"}:
                    raise ValueError("only comma-separated basic geometry cards are supported")
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not nodes or not elements:
        raise ValueError(f"{path}: requires GRID and shell element records")
    vertices = np.asarray(list(nodes.values()), dtype=float) * (0.001 if unit == "mm" else 1.0)
    if not np.isfinite(vertices).all():
        raise ValueError(f"{path}: non-finite GRID coordinates")
    index = {node: i for i, node in enumerate(nodes)}
    triangles, eids, pids = [], [], []
    for element, (pid, ids) in elements.items():
        try:
            corners = [index[key] for key in ids]
        except KeyError as exc:
            raise ValueError(f"{path}: element {element} references absent GRID {exc}") from exc
        triangles.append(corners[:3])
        eids.append(element)
        pids.append(pid)
        if len(corners) == 4:
            triangles.append([corners[i] for i in (0, 2, 3)])
            eids.append(element)
            pids.append(pid)
    return Mesh(vertices, np.asarray(triangles), np.asarray(eids), np.asarray(pids))


def read_element_mapping(path):
    mapping = {}
    for row in _rows(path):
        key = int(row["Elem"])
        if key in mapping:
            raise ValueError(f"{path}: duplicate element {key}")
        mapping[key] = int(row["PID"]), row["Name"]
    return mapping


def read_surface_optics(path):
    """Viewer greybody emissivity and reflectivity by Nastran property ID."""
    optics = {}
    for row in _rows(path):
        pid = int(row["PID"])
        values = float(row["emissivity_3_5um"]), float(row["reflectivity_3_5um"])
        if pid in optics or not all(0 <= value <= 1 for value in values):
            raise ValueError(f"{path}: invalid optical parameters for PID {pid}")
        optics[pid] = values
    return optics


@dataclass(frozen=True)
class ShipFrame:
    mesh_unit: str
    mesh_basis: np.ndarray

    def mesh_to_enu(self, body_to_enu, origin_m, mesh_points_m):
        """Map mesh metres through forward/starboard/up components to ENU."""
        body = orthogonal(body_to_enu, "ship.body_to_enu")
        return np.asarray(mesh_points_m) @ self.mesh_basis @ body.T + vec3(origin_m, "ship.origin_m")

    def mesh_to_enu_at_heading(self, heading_deg, origin_m, mesh_points_m):
        return self.mesh_to_enu(heading_basis(heading_deg), origin_m, mesh_points_m)


class PublicResources:
    """Read the same assets from the bundle or a separately checked-out viewer."""

    def __init__(self, root=PUBLIC_ROOT):
        self.root = Path(root)
        self.manifest = read_json(SOURCE_MANIFEST)

    def path(self, name):
        path = self.root / self.manifest["resources"][name]["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Public resource {name}: {path}")
        return path

    def verify(self):
        for name, record in self.manifest["resources"].items():
            if hashlib.sha256(self.path(name).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Public resource {name} differs from pinned commit {self.manifest['commit']}")

    def ship_frame(self):
        data = read_json(self.path("ship_frame"))
        basis = np.stack([data[f"{axis}_axis_mesh"] for axis in ("forward", "starboard", "up")], axis=-1)
        return ShipFrame(data["mesh_unit"], orthogonal(basis, "ship_frame axes"))

    def mesh(self, name):
        if name not in {"ship_mesh", "sea_mesh"}:
            raise ValueError("mesh name must be ship_mesh or sea_mesh")
        return read_nastran(self.path(name), unit=self.ship_frame().mesh_unit)

    def stack_outlets(self):
        """Outlet centres in mesh metres, with separate normal, area and diameter."""
        outlets = {}
        for item in read_json(self.path("stack_outlets"))["stack_outlets"]:
            centre = vec3(item["center_mesh_mm"], "stack outlet centre") * 0.001
            normal = vec3(item["normal_mesh"], "stack outlet normal")
            diameter, area = float(item["equivalent_diameter_m"]), float(item["area_m2"])
            if not np.isclose(np.linalg.norm(normal), 1) or min(diameter, area) <= 0:
                raise ValueError(f"Invalid stack outlet {item['stack_id']}")
            outlets[item["stack_id"]] = (centre, normal, diameter, area)
        return outlets

    def water_nk(self):
        return spectral_columns(self.path("water_nk"), "wavelength_um", "n", "k")

    def broadband_response(self):
        """Viewer trace. Sort digitised points and average same-wavelength samples."""
        with self.path("broadband_response").open(encoding="latin1", newline="") as stream:
            rows = list(csv.reader(stream))
        if rows[1][1] != "Normalized Photon Response" or any(any(row[2:]) for row in rows[2:]):
            raise ValueError("Expected a single viewer broadband response trace")
        values = np.array([(float(row[0]), float(row[1])) for row in rows[2:] if row and row[0]])
        wave, reverse, counts = np.unique(values[:, 0], return_inverse=True, return_counts=True)
        return check_spectrum(np.column_stack([wave, np.bincount(reverse, values[:, 1]) / counts]), "viewer broadband")

    def _exact_time_spectrum(self, name, utc_time, timestamp_key, fields):
        if not isinstance(utc_time, datetime) or utc_time.utcoffset() != timedelta(0):
            raise ValueError(f"{name}: utc_time must be a UTC datetime")
        meta = read_json(self.path("atmosphere_meta"))
        local_zone = timezone(timedelta(hours=-meta["site"]["timezone_hours_west_of_utc"]))
        local_time = utc_time.astimezone(local_zone).replace(tzinfo=None)
        values = [
            [float(row[field]) for field in fields]
            for row in _rows(self.path(name))
            if datetime.fromisoformat(row[timestamp_key]) == local_time
        ]
        if not values:
            raise ValueError(f"{name}: no exact record at {utc_time.isoformat()}")
        return check_spectrum(values, name)

    def solar_at(self, utc_time):
        return self._exact_time_spectrum("solar_spectrum", utc_time, "Date",
                                         ("wavelength_um", "spectral_irradiance_W_m2_um"))

    def atmosphere_at(self, utc_time, *, height_m, range_m, elevation_deg):
        """Fixed-geometry comparison spectrum with tau, Lpath and Ldown."""
        meta = read_json(self.path("atmosphere_meta"))
        actual = [height_m, range_m, elevation_deg]
        expected = [meta[key] for key in ("height_m", "range_m", "alpha_deg")]
        if not np.allclose(actual, expected, rtol=0, atol=1e-6):
            raise ValueError(f"Viewer atmosphere requires height/range/elevation {expected}; got {actual}")
        if (meta["ldown_semantics"] != "radiance"
                or any(meta[key] != "W m^-2 sr^-1 um^-1" for key in ("lpath_units", "ldown_units"))):
            raise ValueError("Viewer atmosphere radiance-unit mismatch")
        return self._exact_time_spectrum(
            "atmosphere", utc_time, "Time",
            ("lambda_um", "tau", "Lpath_W_m2_sr_um", "Ldown_W_m2_sr_um"),
        )
