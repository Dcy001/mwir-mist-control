# Radiometric features and training

The two channels are ordered `3-5`, then `3.9`. Radiance is band-integrated in W m⁻² sr⁻¹. A radiance tensor is `[2,H,W]`, with Boolean validity `[H,W]` or `[2,H,W]`. Pixel coordinates are `(column, row)`, with integer pixel centres. The `pixel_area_m2` input is the area projected on the calibrated reference plane, in m² per pixel.

## Alignment and quality

Store times as signed integer Unix milliseconds in UTC. Use an explicit continuous record ID and source provenance. Different sources with overlapping times must have distinct IDs. `data.correct_clock` fits an affine correction to source-specific NTP/PPS anchors bracketing the record. `data.align_channels_offline` pairs each band's observations to the supplied 1 Hz grid within a declared tolerance and reports timestamps with at least 20 ms cross-channel jitter as invalid. Supply tolerances appropriate to the recording setup. Retain raw timestamps and returned offsets with quality metadata. `data.keyed_indices` pairs equal record IDs and times for streams already on the common grid. Reject unresolved pairs where exact action or label alignment is required.

`data.outlier_mask` implements IQR fences and the 3σ check per field. Supply the IQR multiplier and the combination rule (`both` or `either`), which Appendix A.1 leaves unspecified. Apply these checks to the relevant continuous source fields, preserve the flagged values in the raw archive and mark them invalid for feature preparation. Online inputs can use `data.causal_indices` with a declared maximum age.

For a decision at time t, the eight-step input ends at t and its target is the action at t+1. Bounded linear repair uses both endpoints available by t and an explicitly configured `max_gap_steps`. The earlier endpoint may precede the eight-step window. In training, nozzle_present=False blocks repair for that nozzle. Time discontinuities and record/subset boundaries also break interpolation. Unresolved terminal gaps leave the window incomplete. Offline preparation caches a repaired vector with its latest endpoint row and admits it only after that row is available. `data.repair_causal_window` applies the same calculation to the observed prefix during prediction and replay.

## Pixel regions and feature order

`roi.build_masks` applies Appendix A.2 to projected stack outlet contours, longitudinally divided side-curtain centrelines and the visible ship waterline. It rasterises the contours with a six-pixel outward offset, extends side centrelines by eight pixels on each side, cuts the bands at their terminal cross-sections and constructs the 20-pixel sea band below the waterline. Camera projections or Boolean masks can be supplied via the session interface. Use calibrated masks where projected segment boundaries differ from these cuts or the waterline is not single-valued.

Appendix A.3 uses sample standard deviation and the field-specific Sea threshold μ + 3σ with an inclusive comparison. A three-by-three cross closing precedes eight-neighbour labelling. Components of at least five pixels contribute their counts and projected areas. Invalid pixels remain outside the statistics and connected regions. Statistics with insufficient valid pixels are marked NaN. The Sea mean for observations uses the observation support. The Sea threshold for residual hotspots uses the paired residual support.

The geometric glint score averages Eq. (2) over the Sea mask. The daytime weighting uses zero direct solar glint at or below the geometric horizon.

Eq. (18) subtracts the model prediction at the actual recorded setting from the observation. Every nozzle in a spray region uses that region's two radiometric feature blocks. `features.FEATURE_NAMES` and `features.FEATURE_UNITS` are the authoritative ordered schema.

| Block | Order and units |
|---|---|
| Environment | Air temperature (°C), visibility (m), 10 m wind speed (m/s), sea temperature (°C) |
| Vessel | Speed (m/s), sine of heading, cosine of heading, roll (°), pitch (°) |
| Actual settings | Pressure (MPa), duty fraction, directional deflection (°) |
| External constraint | Sea sun-glint risk from Eqs. (2)–(3) and Appendix A.4 |
| 3–5 µm, then 3.9 µm | ROI observation mean, ROI residual mean, ROI residual sample standard deviation, ROI residual hotspot count, ROI residual hotspot area (m²), Sea observation mean |

## Prepared training NPZ

All arrays are numeric or Unicode. Load with pickle disabled. Supply rows in increasing timestamp order within each record. Records may interleave. Each must have one source and one phase on the 1 Hz grid. The nozzle axis follows the registry even for records that observe a subset of hardware nozzles.

| Key | Shape and meaning |
|---|---|
| `time_ms`, `record_id`, `source` | `[T]` UTC milliseconds, Unicode record ID and source category `simulation` or `measured` |
| `nozzle_ids` | `[N]` Unicode IDs, matching the registry order |
| `feature_names`, `feature_units`, `action_units` | `[25]`, `[25]` and `[3]`, matching `mwir_control.features` |
| `features` | Floating `[T,N,25]` Eq. (19) features |
| `labels` | Floating `[T,N,3]` external settings labelled at each action timestamp, in MPa, duty fraction, degrees |
| `target_enabled` | Boolean `[T,N]` external label enable state at the label's own timestamp |
| `visibility` | Floating `[T,N]` non-negative MWIR visibility scores used in Table F.5, aligned to the feature rows |
| `label_source`, `visibility_source` | Non-empty Unicode scalars describing each input's origin, including how and when the scores were evaluated |
| `nozzle_present` | Optional Boolean `[T,N]` marking recorded nozzles, defaults to all present |
| `feature_valid` | Optional Boolean `[T,N,25]` quality mask. Mark absent numeric values as NaN |
| `sample_multiplicity`, `sampling_source` | Optional non-negative integer `[T,N]` selection counts keyed by target action time, paired with a Unicode scalar describing the Table F.2 selection |

By default, split distinct UTC timestamps 80/10/10 chronologically within each source (`split_scope=source`), then repair features and construct histories inside each subset and continuous record. This follows the source proportions described in §3.1 at the timestamp level. The combined subsets may overlap in UTC across sources. Resulting window proportions also depend on record lengths, available nozzles, quality and any external selection. The explicit `split_scope=global` option uses one set of time cuts for all records.

Continuous control training uses enabled targets with finite labels within the registered working domain. Observation presence applies to feature history and visibility. An independently supplied target label at t+1 remains eligible if the corresponding t+1 observation is absent. The training report records disabled, missing and out-of-domain targets, incomplete causal histories, and missing visibility. All present nozzles require a score at a window's final input timestamp. The visibility weight uses the maximum available nozzle score at that timestamp, including any supplied scores for unobserved nozzles. Table F.5 leaves the score construction and temporal anchor of `s` open. The interface records the score origin and exposes the chosen anchor.

Use `sample_multiplicity` to provide the Table F.2 stratified selection. Zero excludes an eligible target and a higher count repeats it within its subset. The program counts supplied selections whose windows are ineligible, and reports actual sample counts by source for each subset. Supply the selection from the study records when it is needed to reproduce their proportions. The manuscript gives aggregate coverage and proportions with no joint selection or replacement rule.

Fit the feature z-scores from unique input rows and nozzles available to selected training windows and their temporal predecessors. Fit label z-scores from the selected training targets, including supplied multiplicities. Constant dimensions use unit scale. Dropout of 0.25 is applied to each LSTM output sequence during training. The final state of the third layer feeds the linear three-component readout. Feasibility and temporal penalties use denormalised physical settings as shown in Table F.5. The temporal loss uses the true preceding window of the same nozzle and record, with that earlier decision's own cutoff and eligible enabled target. Its contribution is zero across a disabled interval or when the predecessor is unavailable.

Specify `warmup_start_lr` and `rmse_monitor` (`standardised` or `physical`) since Table F.5 gives the peak and final rates without specifying the ramp start or validation RMSE basis. The five warm-up updates reach 0.001 on epoch five. The cosine schedule reaches 0.00001 at the configured final epoch. Specify `max_gap_steps` for the short-gap repair in Appendix E.1. The best checkpoint records the feature order and units, both normalisations, sources, nozzle IDs, architecture, training options and metrics from the actual run. Prediction uses a version 2 checkpoint, including its gap policy, and returns raw physical settings before projection. Retrain earlier checkpoints with the explicit enable labels and causal preparation.
