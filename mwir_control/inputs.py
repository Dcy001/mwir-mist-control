"""Configuration and explicit NPZ/CSV boundaries for physical evaluation."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .geometry import Camera, TriangleScene, orthogonal, ray_box_interval, solar_unit_enu, vec3
from .resources import (PublicResources, read_element_mapping, read_json, read_nastran, read_surface_optics,
                        sample_spectrum, spectral_columns)


FRAME_UNITS = {
    "radiance": "W m-2 sr-1", "environment": ["degC", "m", "m/s", "degC"],
    "vessel": ["m/s", "deg", "deg", "deg"], "actions": ["MPa", "1", "deg"],
    "pixel_area": "m2", "timestamp": "UTC ms",
}
CHANNELS = ("3-5", "3.9")


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data.items())


def field(mapping, key, scope):
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{scope}: missing field '{key}'")
    return mapping[key]


def absolute(base, path):
    return (base / path).resolve()


def utc_datetime(utc_ms):
    if isinstance(utc_ms, bool) or not isinstance(utc_ms, (int, np.integer)):
        raise ValueError("utc_ms must be an integer UTC epoch timestamp in milliseconds")
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=int(utc_ms))


def nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def action_from_npz(path, registry):
    from .nozzles import Action
    arrays = read_npz(path)
    if tuple(field(arrays, "nozzle_ids", path)) != tuple(registry.ids):
        raise ValueError(f"{path}: nozzle_ids must match the registry order")
    return Action(field(arrays, "values", path), field(arrays, "enabled", path))


def session_registry(path, config=None):
    from .nozzles import NozzleRegistry

    path = Path(path).resolve()
    config = read_json(path) if config is None else config
    return NozzleRegistry.from_rows(read_json(absolute(path.parent, field(config, "nozzles_json", path))))


def load_enable_schedule(session_path, registry, config=None):
    """Load an explicit next-cycle schedule with the time each decision was available."""
    from .cycle import CycleKey, EnablePlan
    from .data import utc_milliseconds

    session_path = Path(session_path).resolve()
    config = read_json(session_path) if config is None else config
    relative = config.get("enable_schedule_npz")
    if not relative:
        raise ValueError("Set session.enable_schedule_npz to an explicit target-cycle enable schedule")
    path = absolute(session_path.parent, relative)
    data = read_npz(path)
    if tuple(field(data, "nozzle_ids", path)) != registry.ids:
        raise ValueError(f"{path}: enable schedule nozzle_ids must match the registry order")
    target = utc_milliseconds(field(data, "target_time_ms", path), "target_time_ms")
    available = utc_milliseconds(field(data, "available_time_ms", path), "available_time_ms")
    if not len(target) or available.shape != target.shape or (target % 1000).any():
        raise ValueError(f"{path}: schedule needs one availability timestamp per exact-second target")
    record = field(data, "record_id", path)
    source = field(data, "source", path)
    for name, values in (("record_id", record), ("source", source)):
        if values.shape != target.shape or values.dtype.kind != "U" or (np.char.strip(values) == "").any():
            raise ValueError(f"{path}: {name} must give a non-empty Unicode value for every target")
    enabled = field(data, "enabled", path)
    if enabled.dtype != np.bool_ or enabled.shape != (len(target), len(registry)):
        raise ValueError(f"{path}: enabled must be Boolean [target cycles, registered nozzles]")
    schedule = {}
    for target_ms, available_ms, record_id, provenance, mask in zip(target, available, record, source, enabled):
        key = CycleKey(str(record_id), int(target_ms) * 1_000_000)
        if key in schedule:
            raise ValueError(f"{path}: duplicate enable target {record_id} at {target_ms} ms")
        if int(available_ms) > int(target_ms) - 1000:
            raise ValueError(f"{path}: enable plan for {record_id} at {target_ms} ms was unavailable at the decision")
        registry.validate_enabled(mask)
        schedule[key] = EnablePlan(key, registry.ids, mask, str(provenance), int(available_ms) * 1_000_000)
    return schedule


@dataclass(frozen=True)
class Frame:
    record_id: str
    utc_ms: int
    source: str
    actual_source: str
    frames: np.ndarray
    valid: np.ndarray
    masks: dict
    glint: float
    environment: np.ndarray
    vessel: np.ndarray
    pixel_area_m2: float
    actual: object
    model: object
    noise: np.ndarray
    data_source: str = None

    def features(self, modelled, actual, registry):
        from .features import build_features
        return build_features(self.frames, modelled, self.valid, self.masks, self.pixel_area_m2,
                              registry.regions, actual.values, self.environment, self.vessel, self.glint)


class SessionInputs:
    """Shared calibration plus per-time frame and boundary inputs."""

    def __init__(self, path):
        from .physics import SpectralGrid

        path = Path(path).resolve()
        self.base = path.parent
        self.config = cfg = read_json(path)
        if cfg.get("schema_version") != 1 or cfg.get("units") != FRAME_UNITS or cfg.get("channels") != list(CHANNELS):
            raise ValueError("Session requires schema_version=1, the documented units and channels ['3-5','3.9']")
        self.public = PublicResources(self.path(cfg["public_root"])) if cfg.get("public_root") else PublicResources()
        self.public.verify()
        self.registry = session_registry(path, cfg)
        self.ship_frame = self.public.ship_frame()
        mesh_config = cfg.get("geometry", {"source": "viewer"})
        if mesh_config["source"] == "viewer":
            self.mesh = self.public.mesh("ship_mesh")
            mapping = read_element_mapping(self.public.path("element_mapping"))
            for eid, pid in zip(self.mesh.element_id, self.mesh.property_id):
                if eid in mapping and mapping[eid][0] != pid:
                    raise ValueError(f"Resource mapping PID differs from ship mesh at element {eid}")
        elif mesh_config["source"] == "calibrated_nastran":
            if cfg["surface_emissivity_source"] != "calibrated":
                raise ValueError("External mesh requires its own calibrated surface emissivity")
            self.mesh = read_nastran(self.path(mesh_config["path"]), unit=mesh_config["unit"])
        else:
            raise ValueError("geometry.source must be viewer or calibrated_nastran")
        sc = field(cfg, "spectral", path)
        wavelength_um = np.asarray(field(sc, "wavelength_um", "spectral"), dtype=float)
        bands_um = np.asarray([[3.0, 5.0], field(sc, "narrow_band_um", "spectral")])
        responses = []
        for key, bounds in zip(("broadband", "narrowband"), bands_um):
            response_cfg = field(sc, key, "spectral")
            if key == "broadband" and response_cfg.get("source") == "viewer_rsr_radiance_weight":
                trace = self.public.broadband_response()
            elif response_cfg.get("source") == "calibrated":
                trace = spectral_columns(self.path(field(response_cfg, "csv", key)), "wavelength_um", "response")
            else:
                raise ValueError(f"spectral.{key}.source requires calibrated response or the explicit viewer broadband option")
            in_band = (wavelength_um >= bounds[0]) & (wavelength_um <= bounds[1])
            response = np.zeros_like(wavelength_um)
            response[in_band] = sample_spectrum(trace, wavelength_um[in_band], key)[:, 0]
            responses.append(response)
        self.spectral = SpectralGrid(wavelength_um, np.stack(responses), bands_um, field(sc, "noise", "spectral"))
        self.wavelength_um = wavelength_um
        self.noise = np.asarray(sc["noise"], dtype=float)
        water = field(cfg, "water_nk", path)
        if water["source"] == "viewer_25degC":
            water_table = self.public.water_nk()
        elif water["source"] == "calibrated":
            water_table = spectral_columns(self.path(water["csv"]), "wavelength_um", "n", "k")
        else:
            raise ValueError("water_nk.source must be viewer_25degC or calibrated")
        n, k = sample_spectrum(water_table, wavelength_um, "water_nk").T
        self.water_nk = n + 1j * k
        self.surface_optics = read_surface_optics(self.public.path("surface_optics"))

    def path(self, path):
        return absolute(self.base, path)

    def _surface_fields(self, arrays, hits):
        kind = self.config["surface_emissivity_source"]
        has_hit = hits.triangle_index >= 0
        if "surface_temperature_k" in arrays:
            temperatures = arrays["surface_temperature_k"]
        elif not has_hit.any():
            temperatures = np.zeros_like(hits.distance_m)
        else:
            temperatures = self._element_field(arrays, hits, "element_temperature_k")
        temperatures = np.asarray(temperatures, dtype=float)
        if temperatures.shape != hits.distance_m.shape or not (
            np.isfinite(temperatures[has_hit]).all() and (temperatures[has_hit] > 0).all()
        ):
            raise ValueError("surface_temperature_k requires positive Kelvin values at every opaque hit")
        # Populate inactive pixels before taking Planck. Their values never enter a source term.
        temperatures = np.where(has_hit, temperatures, 273.15)
        if kind == "viewer_grey":
            emissivity = np.zeros_like(hits.distance_m)
            property_ids = self.mesh.property_id[hits.triangle_index[has_hit]]
            for pid in np.unique(property_ids):
                if int(pid) not in self.surface_optics:
                    raise ValueError(f"No viewer emissivity for property ID {pid}")
            emissivity[has_hit] = [self.surface_optics[int(pid)][0] for pid in property_ids]
        elif kind == "calibrated":
            if "surface_emissivity" in arrays:
                emissivity = arrays["surface_emissivity"]
            elif has_hit.any():
                emissivity = self._element_field(arrays, hits, "element_emissivity")
            else:
                emissivity = np.zeros_like(hits.distance_m)
        else:
            raise ValueError("surface_emissivity_source must be viewer_grey or calibrated")
        emissivity = np.asarray(emissivity, dtype=float)
        shape = hits.distance_m.shape
        if emissivity.shape not in (shape, (*shape, len(self.wavelength_um))):
            raise ValueError("surface_emissivity must be [H,W] or [H,W,L]")
        support = has_hit if emissivity.ndim == 2 else has_hit[..., None]
        return temperatures, np.where(support, emissivity, 0)

    def _element_field(self, arrays, hits, key):
        ids = field(arrays, "element_id", "boundary NPZ")
        values = field(arrays, key, "boundary NPZ")
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != len(ids) or values.ndim == 0 or len(values) != len(ids):
            raise ValueError(f"{key} needs one value per distinct integer element_id")
        indices = {eid: i for i, eid in enumerate(ids)}
        valid = hits.triangle_index >= 0
        eids = self.mesh.element_id[hits.triangle_index[valid]]
        missing = sorted(set(eids) - indices.keys())
        if missing:
            raise ValueError(f"{key} is missing mesh element IDs {missing[:10]}")
        field_values = np.zeros((*hits.distance_m.shape, *values.shape[1:]))
        field_values[valid] = values[[indices[eid] for eid in eids]]
        return field_values

    def _roi(self, cfg, camera, mesh_to_enu, arrays):
        from .roi import build_masks

        if cfg["source"] == "masks":
            ids = tuple(field(arrays, "roi_ids", "frame NPZ"))
            masks = field(arrays, "roi_masks", "frame NPZ")
            if masks.dtype != np.bool_ or masks.shape != (len(ids), *camera.shape) or len(set(ids)) != len(ids):
                raise ValueError("roi_masks must be boolean [R,H,W] with unique roi_ids")
            masks = dict(zip(ids, masks))
        else:
            points = {"pixel_xy": np.asarray, "enu_m": camera.project,
                      "mesh_m": lambda xyz: camera.project(mesh_to_enu(xyz))}
            if cfg["source"] not in points:
                raise ValueError("roi.source must be masks, pixel_xy, enu_m, or mesh_m")
            project = points[cfg["source"]]
            masks = build_masks(camera.shape,
                                {key: project(x) for key, x in cfg["stack_contours"].items()},
                                {key: project(x) for key, x in cfg["side_centrelines"].items()},
                                project(cfg["waterline"]))
        expected = {"Sea", *self.registry.regions}
        if set(masks) != expected:
            raise ValueError(f"ROI identity mismatch: missing {sorted(expected - masks.keys())}, "
                             f"unexpected {sorted(masks.keys() - expected)}")
        return masks

    def _atmosphere(self, cfg, arrays, utc, environment):
        from .physics import Atmosphere, kruse_atmosphere

        source = cfg["source"]
        if source == "kruse":
            path_m = arrays.get("atmosphere_path_m", cfg.get("path_m"))
            if path_m is None:
                raise ValueError("atmosphere.path_m or atmosphere_path_m required for Kruse")
            return kruse_atmosphere(self.wavelength_um, environment[1] / 1000, path_m, environment[0] + 273.15)
        if source == "viewer_table":
            table = self.public.atmosphere_at(utc, height_m=cfg["height_m"], range_m=cfg["range_m"],
                                              elevation_deg=cfg["elevation_deg"])
            tau, path, _ldown = sample_spectrum(table, self.wavelength_um, "viewer atmosphere").T
            return Atmosphere(tau, path, "viewer_table")
        if source == "calibrated_table":
            return Atmosphere(field(arrays, "atmosphere_transmittance", "boundary NPZ"),
                              field(arrays, "atmosphere_path_radiance", "boundary NPZ"), source)
        raise ValueError(f"Unknown atmosphere.source {source!r}")

    def _sea_reflectance(self, cfg, arrays, rays, sun, sea_mask):
        from .physics_spectral import sea_directional_reflectance

        if cfg["source"] == "calibrated_pixels":
            return field(arrays, "sea_reflectance", "boundary NPZ")
        if cfg["source"] != "calibrated_angle_table":
            raise ValueError("sea_reflectance.source must identify calibrated_pixels or calibrated_angle_table")
        result = np.zeros(sea_mask.shape)
        result[sea_mask] = sea_directional_reflectance(
            rays[sea_mask], solar_unit_enu(sun["elevation_deg"], sun["azimuth_deg"]), [0, 0, 1],
            cfg["incident_zenith_deg"], cfg["view_zenith_deg"], cfg["factor"],
        )
        return result

    def load_frame(self, path):
        from .nozzles import Action
        from .physics import Boundary, NozzleCalibration, Plume, RTPCM, RegionCalibration, SpraySystem
        from .roi import glint_weights

        path = self.path(path)
        cfg = read_json(path)
        frame_path = absolute(path.parent, field(cfg, "frame_npz", path))
        arrays = read_npz(frame_path)
        record = nonempty_string(cfg["record_id"], "record_id")
        source = nonempty_string(cfg["source"], "source")
        data_source = cfg.get("data_source")
        if data_source is not None and data_source not in ("simulation", "measured"):
            raise ValueError(f"{path}: data_source must be simulation or measured")
        actual_source = nonempty_string(field(cfg, "actual_source", path), "actual_source")
        stamp = cfg["utc_ms"]
        utc = utc_datetime(stamp)
        if (arrays["utc_ms"].ndim != 0 or arrays["utc_ms"].dtype.kind not in "iu"
                or arrays["utc_ms"].item() != stamp or stamp % 1000 or arrays["record_id"].item() != record):
            raise ValueError(f"{frame_path}: frame identity must match the manifest at an exact UTC second")
        channel_ms = field(arrays, "channel_utc_ms", frame_path)
        if (channel_ms.shape != (2,) or channel_ms.dtype.kind not in "iu"
                or abs(int(channel_ms[0]) - int(channel_ms[1])) >= 20
                or any(abs(int(t) - stamp) > 500 for t in channel_ms)):
            raise ValueError(f"{frame_path}: channel skew must be <20 ms and both frames within the 1 Hz bin")
        if tuple(arrays["channels"]) != CHANNELS or tuple(arrays["nozzle_ids"]) != tuple(self.registry.ids):
            raise ValueError(f"{frame_path}: channel and nozzle identities must be in canonical order")
        for key in ("actual_utc_ms", "state_utc_ms", "boundary_utc_ms"):
            source_stamp = field(arrays, key, frame_path)
            if source_stamp.ndim or source_stamp.dtype.kind not in "iu" or source_stamp.item() != stamp:
                raise ValueError(f"{frame_path}: {key} must match the observation UTC epoch")
        if field(arrays, "actual_record_id", frame_path).item() != record:
            raise ValueError(f"{frame_path}: actual_record_id must match the observation record")
        observation = np.asarray(arrays["observed"], dtype=float)
        valid = arrays["valid"]
        cam = Camera(**cfg["camera"])
        if observation.shape != (2, *cam.shape) or valid.shape != observation.shape or valid.dtype != np.bool_:
            raise ValueError("observed and boolean valid must have shape [2,H,W] matching the camera")
        if not np.isfinite(observation[valid]).all() or (observation[valid] < 0).any():
            raise ValueError("observed radiance must be finite and non-negative on valid pixels")
        actual = Action(arrays["actual_actions"], arrays["actual_enabled"])
        actual_flow = self.registry.flows(actual)
        if "nozzle_health" in arrays and (
            arrays["nozzle_health"].shape != (len(self.registry),)
            or arrays["nozzle_health"].dtype.kind != "U"
        ):
            raise ValueError("nozzle_health must be a Unicode status per registered nozzle")
        environment = np.asarray(cfg["environment"], dtype=float)
        vessel = np.asarray(cfg["vessel"], dtype=float)
        if (environment.shape != (4,) or vessel.shape != (4,) or not np.isfinite([*environment, *vessel]).all()
                or min(environment[[0, 3]]) <= -273.15 or environment[1] <= 0 or environment[2] < 0 or vessel[0] < 0):
            raise ValueError("Invalid aligned environment or vessel input")
        pose = cfg["pose"]
        ship_to_enu = orthogonal(pose["ship_to_enu"], "ship_to_enu", proper=True)
        ship_origin = vec3(pose["ship_origin_m"], "ship_origin_m")

        def mesh_to_enu(points):
            body_to_enu = ship_to_enu * [1, -1, 1]
            return self.ship_frame.mesh_to_enu(body_to_enu, ship_origin, points)

        rays = cam.rays()
        scene = TriangleScene(mesh_to_enu(self.mesh.vertices_m), self.mesh.triangles)
        hits = scene.intersect(cam.origin_m, rays)
        masks = self._roi(cfg["roi"], cam, mesh_to_enu, arrays)
        sea_support = valid & masks["Sea"]
        if (sea_support.sum(axis=(1, 2)) < 2).any():
            raise ValueError("Sea requires at least two valid pixels in each channel")
        sea_means = np.array([observation[k][sea_support[k]].mean() for k in range(2)])
        pixel_area = float(cfg["pixel_area_m2"])
        if not np.isfinite(pixel_area) or pixel_area <= 0:
            raise ValueError("pixel_area_m2 must be a positive projected pixel area")
        sun = cfg["sun"]
        glint_map = np.zeros(cam.shape)
        glint_map[masks["Sea"]] = glint_weights(
            rays[masks["Sea"]], sun["elevation_deg"], sun["azimuth_deg"], environment[2],
        )
        glint = float(glint_map[masks["Sea"]].mean())
        boundary_cfg = cfg["boundary"]
        spectrum = boundary_cfg["solar"]
        if spectrum["source"] == "viewer_spectrum":
            solar = sample_spectrum(self.public.solar_at(utc), self.wavelength_um, "viewer solar")[:, 0]
        elif spectrum["source"] == "calibrated":
            solar = field(arrays, "solar_irradiance", "boundary NPZ")
        else:
            raise ValueError("boundary.solar.source must be calibrated or viewer_spectrum")
        temperature, emissivity = self._surface_fields(arrays, hits)
        plumes = []
        for item in field(boundary_cfg, "plumes", "boundary"):
            entry, exit = ray_box_interval(cam.origin_m, rays, item["lower_enu_m"], item["upper_enu_m"])
            plumes.append(Plume(entry, exit, item["temperatures_k"],
                                field(arrays, item["absorption_key"], frame_path), item["layer_fractions"]))
        atmosphere = self._atmosphere(boundary_cfg["atmosphere"], arrays, utc, environment)
        boundary = Boundary(cam.origin_m, rays, hits.distance_m, temperature, emissivity, masks["Sea"],
                            sea_means, glint_map,
                            self._sea_reflectance(boundary_cfg["sea_reflectance"], arrays, rays, sun, masks["Sea"]),
                            field(boundary_cfg, "sea_emissivity", "boundary"), environment[3] + 273.15,
                            solar, atmosphere, tuple(plumes))
        spray_ref = cfg.get("spray_calibration_json", self.config.get("spray_calibration_json"))
        if not spray_ref and (actual_flow > 0).any():
            raise ValueError("spray_calibration_json required for positive recorded flow")
        spray = None
        if spray_ref:
            spray_cfg = read_json(self.path(spray_ref))
            spray = SpraySystem(ship_to_enu, ship_origin, boundary_cfg["inflow_wind_enu_m_s"],
                                self.water_nk, spray_cfg["water_density_kg_m3"],
                                {key: RegionCalibration(**value) for key, value in spray_cfg["regions"].items()},
                                {key: NozzleCalibration(**value) for key, value in spray_cfg["nozzles"].items()})
        model = RTPCM(self.spectral, boundary, self.registry, spray)
        return Frame(record, stamp, source, actual_source, observation, valid, masks, glint, environment, vessel,
                     pixel_area, actual, model, self.noise, data_source)
