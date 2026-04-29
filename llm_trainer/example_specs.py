from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EXAMPLE_FILES = [
    Path('configs/tasks/move_task.json'),
    Path('configs/tasks/attack_task.json'),
    Path('configs/tasks/aim_fire_task.json'),
    Path('configs/tasks/scout_task.json'),
    Path('configs/tasks/guard_task.json'),
    Path('configs/scenes/move_scene.json'),
    Path('configs/scenes/attack_scene.json'),
    Path('configs/scenes/aim_fire_scene.json'),
    Path('configs/scenes/scout_scene.json'),
    Path('configs/scenes/guard_scene.json'),
]


def main():
    parser = argparse.ArgumentParser(description='Copy the built-in example task and scene configs.')
    parser.add_argument('--out-dir', default=None, help='Optional output directory. Defaults to printing the built-in config paths.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.out_dir is None:
        for rel_path in EXAMPLE_FILES:
            print(repo_root / rel_path)
        return

    out_dir = Path(args.out_dir).resolve()
    for rel_path in EXAMPLE_FILES:
        src = repo_root / rel_path
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f'copied: {dst}')


if __name__ == '__main__':
    main()
