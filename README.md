# Quantum Image Failure Map

Publication-quality Python research repository for:

**“Where Quantum Image Classification Fails: An Experimental Study of Encoding, Spatial Information, Trainability, Measurement Reliability, and Practical Utility”**

This project is a reproducible experimental scaffold for diagnosing failure modes in quantum machine learning for image classification. It does **not** claim quantum advantage and does **not** introduce a new quantum classifier. Instead, it asks a more practical question: where does the quantum image-classification pipeline break down, and how do those breakdowns compare with strong classical baselines?

The repository studies five open problems:

1. Classical image-to-quantum encoding bottlenecks.
2. Barren plateaus and unstable variational training.
3. Loss of spatial image structure.
4. Finite-shot and quantum-noise prediction instability.
5. Lack of clear practical advantage over strong classical baselines.

## Repository Layout

```text
quantum_image_failure_map/
  configs/                 YAML experiment configurations
  scripts/                 Command-line entry points
  src/                     Reusable implementation modules
  src/experiments/         One module per research problem
  tests/                   Unit and smoke-oriented tests
  outputs/                 Timestamped run folders, ignored by git
```

Important implementation modules:

- `src/datasets.py`: Fashion-MNIST and PneumoniaMNIST download/loading, binary filtering, official split handling, deterministic subset indices, split hashes.
- `src/preprocessing.py`: resize, PCA, patch statistics, fitted preprocessing artifacts, train-only fitting discipline.
- `src/encodings.py`: angle and amplitude encoding helpers, feature scaling, resource summaries, fidelity matrices.
- `src/quantum_models.py`: PennyLane variational quantum classifier and parameter-shift gradient support.
- `src/classical_models.py`: logistic regression, linear SVM, RBF-SVM, random forest option, MLP.
- `src/training.py`: reference model training, checkpointing, predictions, gradients, metrics, resource files.
- `src/plotting.py`: plot-only regeneration from saved result files.
- `src/tables.py`: CSV, Markdown, LaTeX, and optional Excel tables.
- `src/verification.py`: run-output integrity checks and verification reports.

## Datasets

- Fashion-MNIST through `torchvision`, using class `0` T-shirt/top and class `6` Shirt, remapped to binary labels.
- PneumoniaMNIST through official `medmnist`, preserving official train, validation, and test partitions.

Preprocessing, PCA, scaling, thresholds, and model selection are fit or chosen using training/validation data only. Test data is reserved for final evaluation.

## Scientific Design

The experiments separate mechanism diagnostics from model comparisons.

For each problem the code records:

- a mechanism metric, such as information retention, gradient magnitude, or shot-induced flip rate;
- a task-performance metric, such as accuracy, F1, AUROC, or calibration error;
- a resource metric, such as circuit depth, circuit evaluations, trainable parameters, wall time, and memory;
- saved raw artifacts so figures can be regenerated without retraining;
- failure indicators used in the unified failure map.

The safeguards implemented in the code and workflow are intentionally conservative:

- no train/test leakage;
- no PCA, scaling, or learned preprocessing fitted on test data;
- no cherry-picking seeds;
- no hiding failed quantum configurations;
- no barren-plateau claim from accuracy alone;
- no quantum-advantage claim from parameter count alone;
- no plot generation that retrains models.

## Installation

Python 3.11 is recommended.

```bash
cd quantum_image_failure_map
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Conda:

```bash
conda env create -f environment.yml
conda activate quantum-image-failure-map
```

On an existing GPU server environment, activate that environment and install only missing packages:

```bash
python - <<'PY'
import importlib.util, subprocess, sys
packages = {
    "pennylane": "pennylane",
    "pennylane_lightning": "pennylane-lightning",
    "medmnist": "medmnist",
}
missing = [pip_name for module, pip_name in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY
```

## Backend Behavior

The runner detects PennyLane backends in this order when `preferred: auto`:

1. `lightning.gpu`
2. `lightning.qubit`
3. `default.qubit`

It records the selected backend and fallback attempts in `environment.json`, `hardware.json`, and `run_summary.json`. GPU is used for PyTorch classical models when available; quantum GPU acceleration is used only when the PennyLane device is installed and operational.

If `lightning.gpu` is not installed or not functional, the code falls back to `lightning.qubit` and then `default.qubit`. The selected backend is always stored in the run metadata.

## Core Commands

Download data:

```bash
python scripts/download_data.py
```

Smoke test:

```bash
python scripts/run_smoke_test.py --config configs/smoke_test.yaml
```

Pilot or full run:

```bash
python scripts/run_all_experiments.py --config configs/default.yaml --mode pilot --resume
python scripts/run_all_experiments.py --config configs/full_experiment.yaml --mode full --resume
```

Run a capped integration check:

```bash
python scripts/run_smoke_test.py --config configs/smoke_test.yaml --max-configs 1
```

Individual problem commands:

```bash
python scripts/run_problem1_encoding.py --config configs/full_experiment.yaml --resume
python scripts/run_problem2_trainability.py --config configs/full_experiment.yaml --resume
python scripts/run_problem3_spatial.py --config configs/full_experiment.yaml --resume
python scripts/run_problem4_reliability.py --config configs/full_experiment.yaml --resume
python scripts/run_problem5_utility.py --config configs/full_experiment.yaml --resume
```

Regenerate outputs from saved data:

```bash
python scripts/aggregate_results.py --run-dir outputs/<run_name>
python scripts/generate_all_plots.py --run-dir outputs/<run_name>
python scripts/generate_tables.py --run-dir outputs/<run_name>
python scripts/verify_outputs.py --run-dir outputs/<run_name>
```

Server example:

```bash
nohup python -u scripts/run_all_experiments.py \
  --config configs/full_experiment.yaml \
  --mode full \
  --resume \
  > server_run.log 2>&1 &
```

Tmux:

```bash
tmux new -s qifm
python -u scripts/run_all_experiments.py --config configs/full_experiment.yaml --mode full --resume
```

GPU monitoring:

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

PBS GPU-server example:

```bash
cat > run_full_gpu.pbs <<'PBS'
#PBS -N qifm_full_gpu
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -o qifm_full_gpu.log

set -e
cd "$PBS_O_WORKDIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate medautopsy

export PYTHONWARNINGS=ignore
export MPLCONFIGDIR="$PWD/.matplotlib"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$MPLCONFIGDIR"

python scripts/run_all_experiments.py \
  --config configs/full_experiment.yaml \
  --mode full \
  --resume \
  --num-workers 4
PBS

qsub run_full_gpu.pbs
```

Check PBS status:

```bash
qstat -u "$USER"
tail -f qifm_full_gpu.log
```

## Output Structure

Every execution creates:

```text
outputs/YYYYMMDD_HHMMSS_<experiment_name>/
  config_resolved.yaml
  command.txt
  environment.json
  git_commit.txt
  hardware.json
  run_summary.json
  run.log
  warnings.log
  checkpoints/
  raw/
  processed/
  metrics/
  predictions/
  gradients/
  resources/
  tables/
  figures/
  figure_data/
  artifacts/
```

Runs are never overwritten. The manifest is stored at `metrics/experiment_manifest.parquet`.

## Generated Outputs

The smoke pipeline is designed to produce:

- at least one saved model checkpoint;
- prediction files with labels, probabilities, predicted labels, sample identifiers, seeds, and configuration IDs;
- initial VQC gradient arrays;
- resource summaries;
- one table set in CSV, Markdown, and LaTeX form;
- one generated figure for each of the five problems;
- a verification report in JSON and Markdown.

Full and pilot runs add more seeds, datasets, preprocessing variants, and experiment grids according to YAML configuration.

## Plot and Table Regeneration

Plotting never trains models. Regenerate figures from saved run data:

```bash
python scripts/generate_all_plots.py --run-dir outputs/<run_name>
```

Regenerate paper tables:

```bash
python scripts/generate_tables.py --run-dir outputs/<run_name>
```

Verify output integrity:

```bash
python scripts/verify_outputs.py --run-dir outputs/<run_name>
```

## Interpretation Guidance

The experiments distinguish mechanism diagnostics from comparative performance. Low VQC accuracy alone is not treated as evidence of a barren plateau; gradient statistics, convergence traces, and failed-run rates are reported separately. Quantum parameter count is not treated as equivalent to classical computational cost.

## Computational Cost Caveats

The full configuration is intentionally conservative but can still be expensive on state-vector simulators. Increase grids in YAML gradually, watch memory usage, and use `--max-configs` for staged server checks. Full finite-shot repeated evaluation multiplies circuit executions by shots, repeats, samples, seeds, and configurations.

## Reproducibility Checklist

- Seeds: `11, 22, 33, 44, 55`; extended set in YAML.
- Python, package, CPU, RAM, CUDA, GPU, and selected backend are recorded.
- Split indices and split hashes are saved.
- Preprocessing artifacts are saved with `joblib` or compressed arrays.
- Predictions, probabilities, gradients, resources, tables, figures, and figure source data are saved.
- Plot generation never trains models.
- Verification writes `artifacts/verification_report.json` and `.md`.

## Limitations

This repository provides a complete, configurable research scaffold and an end-to-end smoke pipeline. Large-scale claims require running the full grids on appropriate hardware and reporting failed or unavailable configurations transparently. Real quantum hardware is optional and not required for the study.

The full study can be computationally expensive. State-vector quantum simulation, parameter-shift gradients, finite-shot repeated evaluation, and noise sweeps scale quickly with samples, qubits, layers, seeds, and shot counts. Start with the smoke configuration, then pilot, then full.

## Development Checks

```bash
python -m compileall src scripts
python -m pytest -q tests
```

The optional PennyLane forward-pass test is skipped automatically if PennyLane is not installed.
