# Integrated SAL benchmark

Files:
- `benchmark_sal_integrated_ml_physics_qpu_vs_classical.py`
- `qpu_runtime_safe_sampler_extended.py`
- `benchmark_sal_integrated_ml_physics_guide.tex`
- `benchmark_sal_integrated_ml_physics_guide.pdf`

## Install

```bash
pip install -r requirements_benchmark_sal_integrated_ml_physics.txt
```

For QPU runs, set your D-Wave token:

```bash
export DWAVE_API_TOKEN=YOUR_TOKEN_HERE
```

## 1) SAL-style classical benchmark on the three-spin dataset

```bash
python benchmark_sal_integrated_ml_physics_qpu_vs_classical.py \
  --models rbm masked_rbm srbm \
  --samplers exact gibbs sa lsb \
  --dataset-source three_spin \
  --n-visible 10 \
  --n-hidden 5 \
  --epochs 100 \
  --negative-samples 512 \
  --output-dir runs_three_spin
```

## 2) Physics benchmark on a J1-J2 transverse-field Ising model

```bash
python benchmark_sal_integrated_ml_physics_qpu_vs_classical.py \
  --models rbm srbm \
  --samplers exact gibbs sa \
  --dataset-source physics_ground \
  --physical-hamiltonian j1j2_1d \
  --j1j2-size 10 \
  --j1 1.0 --j2 0.5 \
  --visible-graph physics \
  --transverse-field 1.0 \
  --epochs 80 \
  --negative-samples 512 \
  --output-dir runs_j1j2
```

## 3) Conservative first QPU run

```bash
python benchmark_sal_integrated_ml_physics_qpu_vs_classical.py \
  --models rbm srbm \
  --samplers sa qpu_pegasus qpu_zephyr \
  --dataset-source physics_ground \
  --physical-hamiltonian long_range_1d \
  --long-range-size 8 \
  --transverse-field 0.8 \
  --epochs 30 \
  --negative-samples 128 \
  --cem-samples 32 \
  --cem-period 5 \
  --annealing-time 20 \
  --num-spin-reversal-transforms 2 \
  --output-dir runs_qpu_start
```

## Main outputs

- `history.csv`: epoch-by-epoch metrics and sampler diagnostics
- `summary.json`: compact summary for each run
- `best_params.npz`, `final_params.npz`: saved parameters
- `figure_<model>_training.png`: training curves
- `figure_final_summary.png`: final comparison figure
