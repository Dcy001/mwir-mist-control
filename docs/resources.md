# Public resources

The bundle pins [viewer commit ebf2581](https://github.com/Dcy001/mwir-ship-signature-viewer/tree/ebf258120d09f28e9177e9c7eb4d8bbc11c97b69). The [manifest](../mwir_control/public/manifest.json) gives the original path and SHA-256 of every file. `mwir-control resources` verifies the assets. A checkout at the same commit can be selected through `public_root`.

| Input | Interpretation |
| --- | --- |
| `model.nas`, `sea_surface_wideY_narrowX_waves_with_wake_minusX.nas` | Free-field Nastran shells in millimetres. The ship supplies opaque ray intersections. The sea mesh supplies reference geometry for preparing the waterline. Equation (14) uses the local mean sea plane. |
| `model_name_elem_mapping_v1.csv` | Component name and PID keyed by element ID. All overlapping entries agree with the Nastran PIDs. The map covers a subset of the mesh. |
| `ship_frame_meta_final.json` | Mesh +X is forward, +Y starboard, +Z up. At heading 0°, forward maps to north and starboard to east. Port quadrants use negative starboard components. |
| `stack_outlet_geometry_final.json` | Outlet centres in mesh millimetres, normals in mesh coordinates, diameters in metres and areas in square metres. Supply the actual CAD contours for Appendix A.2. |
| `ship_surface_optics_3_5um_final.csv` | Viewer grey emissivity and reflectivity, dimensionless by PID. Selecting `viewer_grey` raises on an opaque hit with missing emissivity. PIDs 12 and 13 have no entry. |
| `water_nk_3_5um.csv` | Tabulated real index and absorptive index at 25 °C. Interpolation stays in the supplied 3–5 µm range. |
| `Spectral response curves.csv` | One broadband trace labelled normalised photon response. The reader sorts 110 points and averages duplicate wavelengths to give 93 abscissae. `viewer_rsr_radiance_weight` opts into the viewer convention explicitly. For calibrated camera radiance, supply the measured radiance response. |
| `solar_spectral_irradiance.csv` | Approximate viewer direct solar spectrum in W m⁻² µm⁻¹. Select `viewer_spectrum` for a reference comparison. The manuscript's libRadtran spectrum enters through `calibrated`. |
| `atmos_timevarying_spectral_3_5um_param_regen (1).csv` and metadata | Fixed reference view, 150 m height, 440 m range and 20° elevation. The table contains dimensionless transmittance and path/downwelling spectral radiances in W m⁻² sr⁻¹ µm⁻¹. `viewer_table` uses transmittance and path radiance in Eq. (16). Eq. (14) uses its own solar and sea terms. `kruse` independently calculates Eqs. (12)–(13). |

The viewer time tables have 73 records from 19 July 1984 05:54 to 20 July 1984 05:54 in the documented UTC−04:00 reference time. The reader converts UTC to that fixed offset and requires an exact match. All spectral interfaces specify wavelength in µm. Select external calibrations explicitly when using another acquisition date, viewing geometry or instrument.
