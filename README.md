# 🏎️ Solving CarRacing-v3 with Deep Reinforcement Learning

**Authors**: Adrien Fu, Raphaël Faure  
**Date**: March 2025 (Restructured: November 2026)

---

## 📋 What Is This Project?

This project trains an **autonomous AI agent** to drive a race car around a procedurally-generated track using **Deep Reinforcement Learning**. The car learns entirely from raw pixel observations (96×96 RGB frames) — no hand-coded driving rules.

Two algorithms are compared:

| Algorithm | Implementation | Training Time | Final Score |
|-----------|---------------|--------------|-------------|
| **DQN** (Deep Q-Network) | Custom PyTorch from scratch | ~12 hours | 605+ |
| **PPO** (Proximal Policy Optimization) | Stable-Baselines3 | ~3 hours | 733+ |

---

## 🚀 Quick Start (Fresh Visual Studio Install)

### Step 1: Install Python 3.10+ 
Download from [python.org](https://www.python.org/downloads/). During installation, **check "Add Python to PATH"**.

### Step 2: Install All Dependencies
Open a terminal in this folder and run:
```bash
python setup.py
```
This installs everything: PyTorch, Gymnasium, OpenCV, Box2D, Stable-Baselines3, Jupyter, etc.

### Step 3: Open the Interactive Notebook
```bash
jupyter notebook CarRacing_v3_Complete_Guide.ipynb
```
The notebook walks you through **every step** from environment setup to training to evaluation with rich explanations.

### Step 4: (Optional) Run the Pre-trained Agents Directly
```bash
# Watch the PPO agent drive (opens a Pygame window)
python src/main.py --agent PPO

# Watch the DQN agent drive
python src/main.py --agent DQN

# Headless evaluation (no window, just prints the score)
python src/main.py --agent PPO --no-render
```

### Step 5: (Optional) Launch the Learning Studio
```bash
# Interactive visual training with whole-lap minimap
python src/lap_learning_studio.py --render --episodes 50 --speed 2.0
```

---

## 📁 Project Structure

```
CarRacing_v3_Shareable/
│
├── README.md                              ← You are here
├── setup.py                               ← One-click dependency installer
├── requirements.txt                       ← Pip requirements file
├── CarRacing_v3_Complete_Guide.ipynb       ← 📓 Main notebook (START HERE)
├── CarRacing_v3_Project_Report.pdf         ← 📄 Full PDF report
│
├── src/                                   ← Source code
│   ├── main.py                            ← Run pre-trained agents (DQN/PPO)
│   ├── car_racing_training.py             ← Training pipeline & utilities
│   └── lap_learning_studio.py             ← Advanced training studio with minimap
│
├── models/                                ← Pre-trained model weights
│   ├── true_dqn.pt                        ← Trained DQN model (~1.2 MB)
│   └── true_ppo_data.zip                  ← Trained PPO model (~26 MB)
│
└── assets/                                ← Images, CSVs, reference PDFs
    ├── training_progress.png
    ├── episode_metrics.csv
    ├── debug_*.png                        ← Learning studio screenshots
    └── Report_RL_Fu_Faure.pdf             ← Original research report
```

---

## 🎮 Environment Details

- **Environment**: `CarRacing-v3` from [Gymnasium](https://gymnasium.farama.org/)
- **Physics**: Box2D (tire friction, drift, centrifugal forces)
- **Observation**: 96×96 RGB frames → preprocessed to 60×60 grayscale
- **Actions**: 5 discrete — Nothing, Left, Right, Gas, Brake
- **Reward**: +1000/N per tile, -0.1/frame, -100 off-track

---

## 📚 References

1. Mnih et al. (2013). *Playing Atari with Deep RL* — [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
2. Schulman et al. (2017). *PPO Algorithms* — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
3. Stable-Baselines3 — [GitHub](https://github.com/DLR-RM/stable-baselines3)
4. Gymnasium — [GitHub](https://github.com/Farama-Foundation/Gymnasium)

