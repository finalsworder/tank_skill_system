from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ActionRegistry:
    def __init__(self, root_dir: str | Path = 'actions/trained'):
        self.root_dir = Path(root_dir)

    def action_dir(self, action_id: str) -> Path:
        return self.root_dir / str(action_id)

    @staticmethod
    def _bundle_ready(bundle_dir: Path) -> bool:
        return (
            (bundle_dir / 'model.pt').is_file()
            and (bundle_dir / 'action_spec.json').is_file()
            and (bundle_dir / 'observation_meta.json').is_file()
        )

    def list_actions(self, include_pending: bool = False) -> list[str]:
        if not self.root_dir.exists():
            return []
        actions = []
        for path in self.root_dir.iterdir():
            if not path.is_dir():
                continue
            if include_pending or self._bundle_ready(path / 'best'):
                actions.append(path.name)
        return sorted(actions)

    def bundle_dir(self, action_id: str, bundle: str = 'best') -> str:
        return str(self.action_dir(action_id) / bundle)

    def register_training(
        self,
        action_spec: Any,
        task_config_path: str | Path,
        scene_config_path: str | Path,
        output_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
    ) -> Path:
        action_id = str(getattr(action_spec, 'action_id', action_spec))
        action_dir = Path(output_dir) if output_dir is not None else self.action_dir(action_id)
        action_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            'action_id': action_id,
            'status': 'training',
            'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task_config': str(Path(task_config_path)),
            'scene_config': str(Path(scene_config_path)),
            'output_dir': str(action_dir),
            'run_dir': None if run_dir is None else str(Path(run_dir)),
        }
        (action_dir / 'registration.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return action_dir

    def mark_ready(
        self,
        action_id: str,
        output_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        bundle: str = 'best',
    ) -> None:
        action_dir = Path(output_dir) if output_dir is not None else self.action_dir(action_id)
        registry_file = action_dir / 'registration.json'
        metadata = {}
        if registry_file.exists():
            metadata = json.loads(registry_file.read_text(encoding='utf-8'))
        metadata.update(
            {
                'action_id': str(action_id),
                'status': 'ready' if self._bundle_ready(action_dir / bundle) else 'training',
                'ready_bundle': bundle,
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'output_dir': str(action_dir),
                'run_dir': None if run_dir is None else str(Path(run_dir)),
            }
        )
        registry_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
