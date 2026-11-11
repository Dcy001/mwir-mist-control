# Data and command reference

Use JSON for manifests and NumPy NPZ for aligned arrays. NPZ keys and units are case-sensitive. Store identifiers as Unicode and avoid object arrays. Numeric time keys are UTC epoch times on the stated scale. Boolean masks identify valid data and exact ROI support. Use a separate continuous record ID whenever a recording restarts.

## Align a continuous recording

`mwir-control align config/alignment.json raw-times.npz aligned-indices.npz` processes one continuous recording at a time. Use signed int64 `time_ms` for the requested 1 Hz grid and `wb_time_ms`, `nb_time_ms`, and `<stream>_time_ms` for input clocks. Supply matching `<stream>_anchor_raw_ms` and `<stream>_anchor_utc_ms` arrays to apply NTP/PPS clock correction. Choose the offline tolerance or causal maximum age per stream. Outputs `wb_index`, `nb_index` and each `<stream>_index` map rows back to the source arrays, with `-1` for missing matches. `jitter_ms` and `valid` identify camera pairs with skew below 20 ms. Causal alignment uses samples at or before the query time. Exact timestamp/record joins and Appendix A.1 quality functions are described in [learning.md](learning.md).

Retain the acquisition provenance alongside each processed record, including camera integration time, trigger, FPA temperature, clock anchors, quality decisions, vessel position, and environmental fields used to prepare the model boundary. The core uses the aligned variables specified below.

## Session and camera frames

- Copy `config/session.template.json` and `config/frame.template.json` into the recording directory. Replace `null` entries with measured or calibrated values. In the session, `frames` lists the per-cycle JSON manifests in increasing timestamp order within each record. Frame NPZ paths are relative to the frame JSON. Registry, response, geometry and spray-calibration paths are relative to the session JSON.
- Geometry defaults to the pinned ship mesh. A `calibrated_nastran` geometry object takes `path` and explicit `unit` (`m` or `mm`), uses the same mesh axes and requires calibrated emissivity.
- Each camera specifies `shape=[H,W]`, `fx,fy,cx,cy` in pixels, ENU `origin_m` and the proper rotation `camera_to_enu`. Camera coordinates have +X image right, +Y image down and +Z along the optical axis. Supply calibrated, undistorted and co-registered channel frames.
- Each ship pose specifies the ENU origin and a proper `ship_to_enu` rotation from forward/port/up. The viewer mesh's Y is starboard. The reader transforms the mesh before tracing rays. Nozzle installation positions and rotations use forward/port/up as described in [control.md](control.md). Supply a JSON list of 286 rows following `config/nozzle-row.template.json`.
- `environment` is [air temperature °C, visibility m, wind speed m/s at 10 m, sea temperature °C]. `vessel` is [speed m/s, heading ° clockwise from north, roll °, pitch °]. Solar azimuth uses the same north convention. `pixel_area_m2` is the calibrated projected area of a pixel. Band-integrated radiance and noise use W m⁻² sr⁻¹.
- `roi.source=masks` reads `roi_ids[R]` and Boolean `roi_masks[R,H,W]` from the frame NPZ. Supply exactly `Sea`, `SCC-F`, `SCC-A` and the registered `SC_i` names. To build masks, choose `pixel_xy`, `enu_m` or `mesh_m` and supply `stack_contours` and `side_centrelines` dictionaries keyed by region plus the visible `waterline` polyline. All geometry uses the selected units and frame. Keep the waterline single-valued in the projected image.

Each frame NPZ includes these arrays.

| Key | Shape and meaning |
| --- | --- |
| `utc_ms`, `record_id` | Scalar integer timestamp at an exact UTC second and scalar Unicode record ID, matching the frame manifest |
| `channels`, `channel_utc_ms` | `["3-5","3.9"]` and the two aligned acquisition timestamps. The reader checks the 1 Hz bin and cross-channel skew |
| `observed`, `valid` | Float and Boolean `[2,H,W]`. Valid radiance must be finite and non-negative |
| `actual_utc_ms`, `actual_record_id` | Scalar execution receipt keys matching the observation |
| `state_utc_ms`, `boundary_utc_ms` | Scalar effective timestamps after state and boundary alignment |
| `nozzle_ids`, `actual_actions`, `actual_enabled` | Unicode `[286]`, physical `[286,3]` in MPa, duty fraction and degrees, Boolean `[286]` |
| `nozzle_health` | Optional Unicode `[286]` source health/fault flags. Actual radiometric modelling uses the corresponding measured settings |
| `roi_ids`, `roi_masks` | Required for mask-mode ROIs |

Set the manifest's `source` and `actual_source` to the provenance identifiers of the observation and actual receipt. Set `data_source` to `simulation` or `measured` when preparing training data. For No-mist, disable all nozzles and use zero duty. `forward` exports modelled radiance and Eq. (18) residuals at those actual settings, plus Eq. (1) summaries for the observation, model and residual ordered `[2,R,5]`, with `roi_ids` and `roi_statistics`. `evaluate` takes one or more candidate NPZs, each with `nozzle_ids`, `values[286,3]` and Boolean `enabled[286]`. It evaluates each independently using the declared frame boundary and exports scores, flow and timing.

## Optical and spray boundaries

The session's `spectral.wavelength_um[L]` is strictly increasing and includes 3, 5 and both `narrow_band_um` endpoints around 3.9. The spectral response CSVs require `wavelength_um,response` columns. The calibrated auxiliary channel needs its own response curve. `spectral.noise[2]` gives the NETD-equivalent amplitudes in integrated radiance units. The `viewer_rsr_radiance_weight` option is available for broadband comparison. Water optical indices use the viewer 25 °C table or a calibrated CSV with `wavelength_um,n,k`.

| Frame setting | Required values |
| --- | --- |
| Opaque sources | Supply `surface_temperature_k[H,W]` or `element_id[E]` with `element_temperature_k[E]`. For `surface_emissivity_source=calibrated` supply `surface_emissivity[H,W]` or `[H,W,L]`, or `element_emissivity[E]` or `[E,L]`. All opaque hits need their temperature and emissivity. |
| Plume boxes | Each entry supplies `lower_enu_m`, `upper_enu_m`, `temperatures_k[K]`, `layer_fractions[K]` and an `absorption_key` naming the NPZ array `[K,L]` in m⁻¹. Supply far-to-near layers. Use an empty list if the calibrated boundary has no plume source. |
| Atmosphere | `kruse` takes `path_m` or NPZ `atmosphere_path_m` (scalar or `[H,W]`), and uses the aligned visibility and air temperature. `calibrated_table` reads `atmosphere_transmittance` and `atmosphere_path_radiance` (each `[L]` or `[H,W,L]`). `viewer_table` requires `height_m`, `range_m`, `elevation_deg` and a matching reference timestamp. |
| Sea | Supply `sea_emissivity` as a scalar or `[L]`. `calibrated_pixels` reads the directional factor `sea_reflectance[H,W]`. `calibrated_angle_table` takes `incident_zenith_deg`, `view_zenith_deg` and `factor[incident,view]`. The local mean plane uses ENU up. |
| Solar | `calibrated` takes `solar_irradiance[L]` incident on the mean sea surface, in W m⁻² µm⁻¹. `viewer_spectrum` uses the exact reference timestamp from the public resource. |
| Mist | Use `config/spray-calibration.template.json` with an entry for every required region and nozzle. Specify `inflow_wind_enu_m_s` relative to the vessel. Each per-frame manifest may override the calibration file for the current wind and thermal boundary. See [physics.md](physics.md) for the callable calibration interface. |

## Learning and replay

`prepare-training` joins raw feature rows to an external NPZ with `time_ms[T]`, `record_id[T]`, `nozzle_ids[N]`, `action_units[3]`, `labels[T,N,3]`, Boolean `target_enabled[T,N]` at the label timestamp, `visibility[T,N]`, `label_source` and `visibility_source`. Optional arrays are `nozzle_present[T,N]` for observation presence and the `sample_multiplicity[T,N]` and `sampling_source` pair for stratified selection. Supply a prepared raw-feature NPZ to `train` when extraction takes place upstream. See [learning.md](learning.md) for the schema, causal repair, sample counts and checkpoint compatibility.

### Target-cycle enable schedule

Set `session.enable_schedule_npz` for `predict` and `replay`. Supply `nozzle_ids[N]` in registry order, Boolean `enabled[K,N]`, signed UTC `target_time_ms[K]` and `available_time_ms[K]`, and Unicode `record_id[K]` and `source[K]`. Each target key is unique. A decision at `t` uses the row for its record and target `t+1 s`, with availability at or before `t`. Explicit all-true and all-false rows represent all-on and all-off. Both commands require a schedule row for each requested target. The schedule is distinct from the recorded execution receipt.

`predict` reads `history[T,N,25]` as raw floating features with NaN for missing entries, optional Boolean `feature_valid[T,N,25]`, signed `time_ms[T]` on a continuous 1 Hz grid, and scalar `record_id` and `decision_time_ms`. The last timestamp equals the decision time. Include `nozzle_ids[N]`, `feature_names[25]` and `feature_units[25]`. Replace older prefilled `windows` files with raw histories. The output contains projected `values[N,3]`, `enabled[N]`, raw predictions with NaN in disabled rows, and the record, decision time, target time, enable availability and provenance.

`replay` logs actual settings, the requested enable mask and provenance, the proposal status, virtual projected and final settings, VSO traces, model evaluation counts, flow and measured replay computation time in JSON Lines. Use `--feedback-dir` for actual-state model frames, residuals and raw features. Virtual evaluation uses the latest acquired boundaries. The device adapter receives the target plan and returns an acknowledgement. A subsequent matching receipt supplies the next actual settings.

## Metrics

Copy `config/evaluation.template.json` and include the sections corresponding to your inputs. The metrics entry point emits JSON. Undefined scalars such as R² for a constant target are encoded as `null`.

| Section | Required NPZ keys |
| --- | --- |
| `prediction` | `timestamps_ns[T]`, `nozzle_ids[286]`, `action_units[3]`, `predicted[T,286,3]`, `labels[T,286,3]`, Boolean `target_enabled[T,286]`, `prediction_source`, `label_source` |
| `comparison` trials | `timestamps_ns[T]`, `grid_id`, `source`, `channels[2]`, `radiance_units="W m-2 sr-1"`, `radiance[T,2,H,W]`, Boolean `valid[T,2,H,W]` or `[T,H,W]`. Temperature metrics use `temperature[T,H,W]` and `temperature_units` (`K` or `degC`) in every trial. |
| `comparison` support | `timestamps_ns[T]`, `grid_id`, `roi_ids[R]`, `roi_masks[R,H,W]` or `[R,T,H,W]`, `pixel_area_m2` scalar or `[T]`, `noise[2]`, `glint[T]`. Full scope requires exactly `Sea`, `Global`, both SCC regions and `SC_1` through `SC_278`. Declare local midnight in UTC epoch ns and the H.7 `epsilon` in the JSON. |
| `water` | `timestamps_ns[T]` on a continuous 1 Hz grid, `source`, `full_load_source`, `flow_units="L/min"`, `nozzle_ids[286]`, `flow_l_min[T,286]`, `full_load_l_min[286]` or `[86400,286]` |
| `latency` | `timestamps_ns[T]`, `source`, `latency_units="ms"` and `io,phys,prep,lstm,vso,guard,tx` arrays of length T from actual per-cycle timing |

For a partial comparison, explicitly set `comparison.roi_scope` to `{"mode":"subset","subset_id":"survey-a","roi_ids":["Sea","SC_1"]}` or list the canonical regions available in that survey. Supply `Sea` and at least one spray ROI, and match that list exactly in the support NPZ. The export identifies the participating regions and coverage and labels the subset visibility maxima separately. See [control.md](control.md) for metric windows and aggregation rules.
