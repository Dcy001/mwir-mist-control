# Physical forward model

`RTPCM` prepares the source, atmosphere and sea terms for one radiometric cycle. Call `evaluate(action)` with an actual action or a feasible candidate in registry order. Its `frames` array is ordered `[3–5 μm, 3.9 μm]` and has shape `[2,H,W]` and units W m⁻² sr⁻¹. It also returns the assigned curtain index (`-1` outside mist), LOS chord length in metres, the region index order, the atmosphere source and a count of negative spectral samples in the fitted background.

| Manuscript | Implementation |
| --- | --- |
| Appendix B.1–B.2 | Camera rays and nearest opaque facet and plume box intersections in `geometry`; mutually exclusive masks in `physics_spectral.classify_los` |
| Eqs. (5)–(7), Appendix B.3–B.4 | Greybody surface emission and background reflection, layerwise Schwarzschild plume emission, Planck radiance and the two-channel Sea background inversion |
| Appendix C.1–C.4 | Mount, deflection and vessel rotations, quadratic centrelines, circular SCC and thin SC envelopes, LOS chord, nearest region and foreground screening |
| Appendix C.5–C.6, Eqs. (8)–(11) | Calibrated flow curves, regional flow, liquid water content, lognormal effective radius, droplet number density, spherical Mie coefficients, attenuation and orientation-weighted path correction |
| Appendix D, Eqs. (12)–(17) | Kruse or explicit tabulated atmospheric path, sea emission and glint-modulated BRDF, and separate response-weighted band integrations |

Wavelengths and spectral quadrature use micrometres. Planck converts from per metre to per micrometre internally. Temperatures used by the radiometry are absolute kelvin. In the Kruse path the visibility is in kilometres, the supplied path in metres and the extinction coefficient is in km⁻¹. Supply the camera noise floor separately for each band in the units of the integrated radiance. Choose a grid containing the boundaries of both measured response curves. Response values are radiance weights, with `integrate` using their absolute scale. Eq. (D.2) uses response-weighted *spectral density* of solar irradiance, while Eq. (17) integrates without dividing by response area.

## Numerical conventions and calibration inputs

### Numerical conventions

- For Eq. (C.14), the relative speed is the norm of the arithmetic mean of the active regional jet-velocity vectors minus the relative inflow vector. The arithmetic mean is the implementation's aggregation convention for SCC. Hydraulic flow is converted from L/min to m³/s.
- The solver interprets the swept sections in Eqs. (C.7)–(C.8) as lying in planes through each centreline station orthogonal to the regional axis. It rejects centrelines that fold back along that axis. The LOS interval keeps the first entry and last exit in Eq. (C.9). An opaque or plume intersection strictly before entry screens the interval under Eq. (C.12). Curtain ties use registry order. Each SCC group must be wholly enabled or disabled.
- Among plume boxes with the same nearest entry, the first supplied box is selected. An opaque hit tied with the nearest plume entry takes priority under Appendix B.2.
- Spherical Mie coefficients are evaluated at the lognormal effective radius used by the extinction cross-section and number-density closure. This single-radius calculation is the implementation's optical convention. The orientation kernel follows Eq. (C.18).
- Appendix B.4 uses an unconstrained two-basis fit. The model retains signed spectral samples and reports their count. Eq. (15) adds the explicit sea term inside the Sea ROI.

### Calibration inputs

Use an ENU world frame and a right-handed nozzle vessel frame (+X forward, +Y port, +Z up). Supply the installation point in metres, the mount rotation from local width/thickness/spray axes to vessel axes, and the deflection axis in mount coordinates. Eq. (C.1) uses candidate deflection in degrees. Supply the relative inflow wind in ENU m/s.

Supply nozzle pressure in MPa, jet speed in m/s, ENU centreline curvature in m⁻¹ and effective length in metres. Pressure tables interpolate within their calibrated span and reject out-of-range values. Region calibration supplies length and initial half-thickness in metres, dimensionless growth, number-median diameter in metres and geometric standard deviation. SC uses the Table F.1 fan width of 1.27 m and reference distance of 0.508 m unless a calibration supplies both. A deterministic Python provider can supply nozzle or region calibration as a function of action and wind. The model reports missing calibration by nozzle or region.

The boundary supplies opaque-hit temperatures and emissivities, plume layers, the Sea mask and two observed band means, sea emissivity and temperature, directional reflectance and incident solar irradiance. Plume temperatures and absorption coefficients (m⁻¹) use far-to-near layer order, with LOS thickness fractions summing to one. The sea reflectance helper evaluates a measured incident/emergent zenith grid within its coverage. Use Kruse for 6–50 km visibility or supply a provenance-labelled atmosphere table with spectral path radiance and transmittance. See [input-contract.md](input-contract.md) for the file schema.

