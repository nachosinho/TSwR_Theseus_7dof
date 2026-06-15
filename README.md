# Theseus 7DoF

Differentiable motion planning for the Franka Panda 7-DoF manipulator using Theseus and a learned neural warm-start model.

## Authors:

Sebastian Nachowiak, Marcin Osztynowicz
Poznan Univeristy of Technology, 2026

## Features

* Franka Panda 7-DoF motion planning
* Differentiable GPMP2 optimization in Theseus
* Linear warm-start baseline
* Neural MLP warm-start
* Collision-aware trajectory optimization
* Benchmarking and comparison tools

## Requirements

* Python 3.10
* PyTorch
* Theseus
* PyBullet
* NumPy
* SciPy
* Matplotlib

## Install Theseus

```bash
git clone https://github.com/facebookresearch/theseus.git
cd theseus

conda create -n theseus_env python=3.10
conda activate theseus_env

pip install -e .
```

## Install this project

```bash
git clone https://github.com/nachosinho/TSwR_Theseus_7dof.git
cd TSwR_Theseus_7dof

pip install -r requirements.txt
```

## Training

Train the neural warm-start model:

```bash
python train_dgpmp2_generalization.py
```

Useful options:

```bash
--epochs
--inner-iters
--train-steps
--points-per-link
--hidden-dim
--lr
--device
--checkpoint
```

Example:

```bash
python train_dgpmp2_generalization.py \
    --epochs 30 \
    --inner-iters 5 \
    --device cpu \
    --checkpoint warmstart_mlp_unroll.pt
```

## Motion Planning

Run planner with linear initialization:

```bash
python run_planner.py
```

Run planner with neural warm-start:

```bash
python run_planner.py \
    --use-mlp-warmstart \
    --mlp-checkpoint warmstart_mlp_unroll.pt
```

Common options:

```bash
--scenario
--max-iterations
--use-mlp-warmstart
--mlp-checkpoint
```

Example:

```bash
python run_planner.py \
    --scenario s8_clutter \
    --use-mlp-warmstart \
    --mlp-checkpoint warmstart_mlp_unroll.pt \
    --max-iterations 30
```

## Trajectory Validation

Validate a saved trajectory:

```bash
python validate_trajectory.py
```

## Benchmarking

The script:

```bash
bench.sh
```

automatically executes benchmark experiments for selected scenarios and stores:

* optimization times,
* convergence statistics,
* collision margins,
* generated plots,
* benchmark_result.txt files.

Run:

```bash
chmod +x bench.sh
./bench.sh
```

## Scenarios

Available benchmark environments include:

* s2_central_obstacle
* s4_goal_near_obstacle
* s5_around_back
* s7_low_reach
* s8_clutter

## Repository

https://github.com/nachosinho/TSwR_Theseus_7dof

