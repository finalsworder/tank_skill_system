# Tank Skill System v3

This project trains and evaluates reusable tank combat skills with a discrete PPO policy and a rule-scripted battle environment.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train A Skill

```powershell
python train_skill.py `
  --action-spec configs/tasks/move_task.json `
  --scene-config configs/scenes/move_scene.json `
  --output-dir actions/trained/move `
  --num-envs 8 `
  --episodes-per-env 200 `
  --max-time 200
```

## Run A Demo

```powershell
python demo_scripted_3v3.py
```

Training outputs, checkpoints, logs, local IDE files, and cache files are intentionally ignored by git. Keep source code and JSON configs in the repository; keep trained model bundles local unless you explicitly want to publish them separately.
