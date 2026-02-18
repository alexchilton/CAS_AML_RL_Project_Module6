# CAS AML Module 6 — Reinforcement Learning Project
**University of Bern — CAS in Advanced Machine Learning**

A multi-environment reinforcement learning project exploring classic tabular methods, deep RL, recurrent architectures, and genetic algorithms across three progressively complex custom environments.

---

## Project Structure

```
├── frozen_lake/        # Part 1 – Frozen Lake variants & Bern→Mürren route problem
├── roads_muerren/      # Part 2 – Custom graph navigation environment
└── battle_rpg/         # Part 3 – Turn-based RPG with full graphical interface
```

---

## Part 1 – Frozen Lake & Route Optimisation (`frozen_lake/`)

Starting from OpenAI Gymnasium's [FrozenLake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) environment, this section explores:

- **Q-table agents** on 4×4 and 8×8 grids, slippery and non-slippery variants
- **DQN agent** for the same environments
- **Custom Bern→Mürren route problem** (`01.bern_2_muerren.ipynb`): a graph-based navigation environment modelling the Swiss rail/road network, solved with RL agents

---

## Part 2 – Roads & Mürren Navigation (`roads_muerren/`)

Custom graph environment representing the Bern–Mürren travel network. Includes data creation notebooks and rendered outputs (HTML, SVG, MP4 visualisations).

---

## Part 3 – Battle RPG (`battle_rpg/`)

The main project: a fully custom **turn-based RPG combat environment** implemented as a [Gymnasium](https://gymnasium.farama.org/) environment, complete with a PyGame graphical interface.

### Environment

A knight must defeat two bandits using a combination of attacks and potions:

| Entity | Max HP | Strength | Potions |
|--------|--------|----------|---------|
| Knight (agent) | 30 | 10 (±5 random) | 3 |
| Bandit 1 | 20 | 6 (±3 random) | 1 |
| Bandit 2 | 20 | 6 (±3 random) | 1 |

**Action space:** `{0: attack bandit 1, 1: attack bandit 2, 2: use potion}`  
**Observation space:** HP values, potion counts, last actions taken, valid action mask (13-dimensional)

### Algorithms Implemented

| Algorithm | File | Notes |
|-----------|------|-------|
| DQN | `graphic_rpg_rl.py` | Standard deep Q-network with experience replay |
| Q-Learning | `graphic_rpg_rl_qlearning.py` | Tabular baseline |
| GRU-DQN | `gru_dqn_rpg.py` / `gru_enhanced.py` | Recurrent DQN — adds memory over combat history |
| Genetic Algorithm | `genetic_rpg_rl.py` / `simple_genetic.py` | Policy evolution without gradient descent |

### PyGame Visualiser

The full graphical interface (`graphic_rpg.py`, `graphic_env.py`, `graphic_visualizer.py`) renders:
- Animated combat between knight and bandits
- HP bars, potion counts, action history
- Main menu, intro screens, and per-episode metrics

### Pre-trained Models

Several trained model checkpoints are included (`.zip` files):
- `graphic_rpg_model_best.zip` — best DQN checkpoint
- `recurrent_rpg_model.zip` — best GRU-DQN checkpoint
- `genetic_rpg_model_best.pkl` — best genetic algorithm policy

### Training Metrics

Training logs and metric plots (`training_log.txt`, `agent_performance_results.png`, `genetic_training_metrics.png`) are included for comparison across methods.

---

## Stack

- Python 3.x
- [PyTorch](https://pytorch.org/) 2.1
- [Gymnasium](https://gymnasium.farama.org/) 0.29
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) 2.2
- [PyGame](https://www.pygame.org/) 2.5
- NumPy · Matplotlib · NetworkX

## Installation

```bash
pip install -r battle_rpg/requirements.txt
```

## Running the graphical RPG

```bash
cd battle_rpg
python graphic_rpg.py
```

To train from scratch:

```bash
python graphic_rpg_rl.py        # DQN
python gru_dqn_rpg.py           # GRU-DQN
python genetic_rpg_rl.py        # Genetic algorithm
```
