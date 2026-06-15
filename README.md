# Panda dGPMP2 Generalization

Projekt przedstawia generalizację dGPMP2 dla manipulatora Franka Panda 7 DoF z wykorzystaniem sieci MLP do neural warm-start.

## Requirements

- Python 3.10
- CUDA (opcjonalnie)

## Install Theseus

```bash
git clone https://github.com/facebookresearch/theseus.git
cd theseus

conda create -n theseus_env python=3.10
conda activate theseus_env

pip install -e .
```

## Install project

```bash
git clone https://github.com/nachosinho/Panda-dGPMP2-Generalization.git
cd Panda-dGPMP2-Generalization

pip install -r requirements.txt
```

## Training

```bash
python train_dgpmp2_generalization.py
```

## Evaluation

```bash
python run_planner.py
```

## Trajectory validation

```bash
python validate_trajectory.py
```
