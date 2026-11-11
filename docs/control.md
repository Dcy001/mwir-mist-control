# Control and evaluation

## Nozzle registry and hydraulics

Keep the same ordered nozzle IDs in the registry, observations, actual settings, training labels, checkpoints and command receipts. Supply 5 SCC-F, 3 SCC-A and 278 SC nozzles. SCC-F comprises two Port, two Starboard and one Midships nozzle. SCC-A comprises one of each. The registry checks 139 SC nozzles on each side and checks one nozzle per `SC_i` ROI (`SC_1` through `SC_278`).

Each registry row has these fields.

| Field | Meaning |
| --- | --- |
| `nozzle_id` | Stable hardware ID |
| `region` | SCC-F, SCC-A or the assigned SC_i ROI |
| `kind` | `SCC-port`, `SCC-starboard`, `SCC-midships` or `SC` |
| `position_m` | Installation point in the right-handed ship frame, metres. +X points forward, +Y port and +Z up. Convert viewer mesh positions from mm and flip mesh Y. |
| `mount_rotation` | 3 × 3 proper rotation from nozzle-local coordinates to the right-handed ship frame. Its columns are local width, thickness and reference spray axes. |
| `deflection_axis` | Unit rotation axis in nozzle-local coordinates |
| `reference_pressure_mpa` | Reference working pressure for the water term in Eq. (25), at unit duty |
| `flow_extension` | Calibrated [MPa, L/min] unit-duty pairs outside the Table F.1 pressure interval, when needed |

`Action` uses a [N,3] array in MPa, duty fraction and degrees, with a Boolean enable mask. A disabled nozzle has zero duty and flow. Candidate projection writes zero in all three disabled settings. Each SCC group must be entirely enabled or disabled. Actual receipts preserve measured settings, including departures from commanded working settings.

Appendix C.5 uses piecewise linear flow interpolation. The two stack totals in Table F.1 give `q_PS = (Q_F - Q_A)/2` and `q_M = Q_A - 2 q_PS`. The SC table already gives single-nozzle flow. Registry flow outputs are duty-averaged L/min. A 1 s cycle uses `q/60` L and Eq. (H.14) converts L/min × s to m³ by dividing by 60,000.

SCC published flow points end at 0.827 MPa and SC points start at 0.20 MPa. These leave parts of their working domains uncovered. Supply measured anchors covering candidate and actual pressures in those ranges. The interpolation raises a field-specific error for any uncovered pressure.

## Candidate selection and feedback

`visibility` implements Eqs. (23)–(24) over co-registered bands. Its channel noise terms are NETD-equivalent radiance amplitudes in the same units as the band-integrated frames. Each scored ROI needs valid pixels in both channels and the Sea ROI needs at least two. Replay scores regions with enabled nozzles. Complete-study evaluation checks the full ROI list separately.

`select_action` projects the LSTM outputs for enabled nozzles, evaluates the projected set, then searches enabled nozzles whose regions exceed 3. Each probe substitutes one enabled nozzle into the projected set. The finite difference uses two feasible samples, centred in the interior and one-sided at a bound. Each local search compares the initial candidate with the iterates in Eq. (26). The final combined setting gets its own forward evaluation when it differs from the projected set. The selected settings follow Eq. (27).

`CycleController.ingest` requires an observation and actual receipt with the same continuous record and timestamp. It computes model radiance at the actual setting before writing the residual into the history. `propose(enable_plan)` checks the target key, availability time and nozzle order, then uses the latest acquired boundary for virtual evaluation. Enabled nozzles require a complete eight-step window after causal short-gap repair. A missing plan or incomplete enabled window defers the proposal. An explicit all-off plan bypasses LSTM inference.

The device adapter implements `issue(plan) -> CommandAck` and receives the selected settings with their enable mask. A separate `ActualReceipt` carries the settings used for the next observation. Reconcile uncertain acknowledgements against the hardware before retrying. The Python `replay` API accepts observation and receipt pairs and a mapping from target cycle keys to enable plans. `cycle_log` records enable provenance, candidates, costs and measured phase timings. Complete end-to-end timing requires an acquisition measurement and a device acknowledgement.

## Appendix H metrics

- `prediction_metrics` reports MAE, RMSE, R² and CVR per nozzle and parameter, with counts of paired enabled targets. Constant labels make R² undefined.
- `compare_trials` uses exactly matched timestamps, a shared grid identity and the intersection of valid radiance pixels for No-mist and all supplied strategies. Full evaluation requires the complete canonical ROI list and valid support in each required window. Explicit subsets report their coverage. Temperature comparisons use the intersection of finite temperature samples. Supply local midnight to select the Appendix H.2 windows and the positive H.7 epsilon for visibility ratios. The 3.9 µm calculation rounds the top 1% pixel count up, uses the full projected ROI area in Eq. (H.10) and averages the specified per-frame statistics.
- `water_metrics` accepts aligned nozzle flow records in L/min. Supply the 24 h full-load reference as a constant unit-duty flow vector or an aligned 1 Hz schedule. The result uses that entire 24 h denominator, even when the measured use covers a shorter period.
- `latency_metrics` accepts aligned per-cycle timings in milliseconds for `io`, `phys`, `prep`, `lstm`, `vso`, `guard` and `tx`. Deadline misses are strictly above 1,000 ms, with population standard deviation for jitter.
