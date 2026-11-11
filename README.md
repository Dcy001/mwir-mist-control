# Radiance-guided MWIR mist control

Python implementation of the RT-PCM, shared LSTM and visibility-triggered VSO in *Radiance-guided spatiotemporal water-mist control enables adaptive mid-wave infrared stealth for ships*. The package covers two-channel radiometric processing, the 286-nozzle forward model and controller, recorded-action replay, and evaluation.

Simulation scenarios, data generation, the [MWIR Ship Signature Viewer](https://github.com/Dcy001/mwir-ship-signature-viewer) and the §3.6 land-based experiment are maintained in separate projects. This package reads selected geometry and spectral resources from the pinned viewer release.

## Install

Use Python 3.9–3.12 in a virtual environment. Install with training and test dependencies.

```sh
python -m pip install ".[test]"
mwir-control --help
python -m pytest -q
```

For a physical-model-only installation, use `python -m pip install .`. The tested dependency versions are listed in [requirements-verified.txt](requirements-verified.txt).

## Connect data

Start with [config/session.template.json](config/session.template.json) and [config/frame.template.json](config/frame.template.json). Follow the [input contract](docs/input-contract.md) for recordings, calibration, labels and the next-cycle nozzle enable schedule. The [resource inventory](docs/resources.md) records the source commit and coverage of the bundled inputs.

## Run

```sh
mwir-control resources
mwir-control align local/alignment.json local/raw-times.npz out/aligned-indices.npz
mwir-control validate local/session.json
mwir-control forward local/session.json local/frames/f0.json out/frame.npz
mwir-control evaluate local/session.json local/frames/f0.json out/candidates.json candidate-a.npz candidate-b.npz
mwir-control prepare-training local/session.json local/labels.npz out/training.npz
mwir-control train local/session.json out/training.npz out/model.pt --split-scope source --warmup-start-lr 0 --rmse-monitor standardised --max-gap-steps 1
mwir-control predict local/session.json out/model.pt local/history.npz out/actions.npz
mwir-control replay local/session.json out/model.pt out/replay.jsonl
mwir-control metrics local/evaluation.json out/metrics.json
```

Paths in a session manifest are relative to its directory. `--split-scope source` follows §3.1 and splits simulation and measured data separately in chronological order. The warm-up start of zero, standardised validation RMSE and one-step gap limit are example settings. Set these three options from your training protocol.

Replay matches observations to recorded execution receipts and evaluates proposals against the enable schedule available at each decision time. The device interface in `cycle.py` accepts an implementation supplied by the operator.
