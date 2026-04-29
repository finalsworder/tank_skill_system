from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


def _normalize_obstacle(data: Any) -> Dict[str, float]:
    if isinstance(data, dict):
        if all(key in data for key in ('x', 'y', 'w', 'h')):
            return {
                'x': float(data['x']),
                'y': float(data['y']),
                'w': float(data['w']),
                'h': float(data['h']),
            }
        if all(key in data for key in ('center_x', 'center_y', 'size')):
            size = float(data['size'])
            return {
                'x': float(data['center_x']) - size / 2.0,
                'y': float(data['center_y']) - size / 2.0,
                'w': size,
                'h': size,
            }
    if isinstance(data, (list, tuple)) and len(data) == 3:
        center_x, center_y, size = [float(x) for x in data]
        return {
            'x': center_x - size / 2.0,
            'y': center_y - size / 2.0,
            'w': size,
            'h': size,
        }
    raise ValueError(f'Unsupported obstacle format: {data!r}')


@dataclass
class ValueSpec:
    mode: str = 'none'
    fixed: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[int]] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ValueSpec':
        return ValueSpec(
            mode=data.get('mode', 'none'),
            fixed=data.get('fixed'),
            low=data.get('low'),
            high=data.get('high'),
            choices=data.get('choices'),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpawnRegion:
    mode: str
    rect: Optional[List[float]] = None
    point: Optional[List[float]] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SpawnRegion':
        return SpawnRegion(mode=data['mode'], rect=data.get('rect'), point=data.get('point'))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnitPolicySpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'UnitPolicySpec':
        return UnitPolicySpec(name=data['name'], params=dict(data.get('params', {})))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationConfig:
    radar_num_rays: int = 72
    radar_max_distance: float = 400.0
    radar_body_relative: bool = True
    max_allies: int = 2
    max_enemies: int = 3

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ObservationConfig':
        return ObservationConfig(
            radar_num_rays=int(data.get('radar_num_rays', 72)),
            radar_max_distance=float(data.get('radar_max_distance', 400.0)),
            radar_body_relative=bool(data.get('radar_body_relative', True)),
            max_allies=int(data.get('max_allies', 2)),
            max_enemies=int(data.get('max_enemies', 3)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskSlotSpec:
    target_x: ValueSpec = field(default_factory=ValueSpec)
    target_y: ValueSpec = field(default_factory=ValueSpec)
    ally_id: ValueSpec = field(default_factory=ValueSpec)
    enemy_id: ValueSpec = field(default_factory=ValueSpec)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TaskSlotSpec':
        return TaskSlotSpec(
            target_x=ValueSpec.from_dict(data.get('target_x', {'mode': 'none'})),
            target_y=ValueSpec.from_dict(data.get('target_y', {'mode': 'none'})),
            ally_id=ValueSpec.from_dict(data.get('ally_id', {'mode': 'none'})),
            enemy_id=ValueSpec.from_dict(data.get('enemy_id', {'mode': 'none'})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_x': self.target_x.to_dict(),
            'target_y': self.target_y.to_dict(),
            'ally_id': self.ally_id.to_dict(),
            'enemy_id': self.enemy_id.to_dict(),
        }


@dataclass
class RewardWeights:
    weights: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'RewardWeights':
        return RewardWeights(weights={str(k): float(v) for k, v in data.items()})

    def to_dict(self) -> Dict[str, float]:
        return dict(self.weights)


@dataclass
class IntelConfig:
    reveal_selected_enemy_on_reset: bool = False
    share_team_memory: bool = True

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'IntelConfig':
        return IntelConfig(
            reveal_selected_enemy_on_reset=bool(data.get('reveal_selected_enemy_on_reset', False)),
            share_team_memory=bool(data.get('share_team_memory', True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TerminationSpec:
    success: List[str] = field(default_factory=list)
    failure: List[str] = field(default_factory=list)
    max_steps: int = 200
    reach_radius: float = 35.0
    hold_steps: int = 1

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TerminationSpec':
        return TerminationSpec(
            success=list(data.get('success', [])),
            failure=list(data.get('failure', [])),
            max_steps=int(data.get('max_steps', 200)),
            reach_radius=float(data.get('reach_radius', 35.0)),
            hold_steps=int(data.get('hold_steps', 1)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResetTemplate:
    template_id: str
    map_size: List[float]
    obstacles: List[Dict[str, float]]
    red_spawns: List[SpawnRegion]
    blue_spawns: List[SpawnRegion]
    red_count: int
    blue_count: int
    red_heading_range: List[float] = field(default_factory=lambda: [0.0, 360.0])
    blue_heading_range: List[float] = field(default_factory=lambda: [0.0, 360.0])
    initial_known_enemy_ids: List[int] = field(default_factory=list)
    ally_policies: Dict[str, UnitPolicySpec] = field(default_factory=dict)
    enemy_policies: Dict[str, UnitPolicySpec] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ResetTemplate':
        return ResetTemplate(
            template_id=str(data['template_id']),
            map_size=[
                float(data.get('map_size', [1000.0, 1000.0])[0]),
                float(data.get('map_size', [1000.0, 1000.0])[1]),
            ],
            obstacles=[_normalize_obstacle(o) for o in data.get('obstacles', [])],
            red_spawns=[SpawnRegion.from_dict(x) for x in data.get('red_spawns', [])],
            blue_spawns=[SpawnRegion.from_dict(x) for x in data.get('blue_spawns', [])],
            red_count=int(data['red_count']),
            blue_count=int(data['blue_count']),
            red_heading_range=[
                float(data.get('red_heading_range', [0.0, 360.0])[0]),
                float(data.get('red_heading_range', [0.0, 360.0])[1]),
            ],
            blue_heading_range=[
                float(data.get('blue_heading_range', [0.0, 360.0])[0]),
                float(data.get('blue_heading_range', [0.0, 360.0])[1]),
            ],
            initial_known_enemy_ids=[int(x) for x in data.get('initial_known_enemy_ids', [])],
            ally_policies={str(k): UnitPolicySpec.from_dict(v) for k, v in data.get('ally_policies', {}).items()},
            enemy_policies={str(k): UnitPolicySpec.from_dict(v) for k, v in data.get('enemy_policies', {}).items()},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'template_id': self.template_id,
            'map_size': self.map_size,
            'obstacles': self.obstacles,
            'red_spawns': [x.to_dict() for x in self.red_spawns],
            'blue_spawns': [x.to_dict() for x in self.blue_spawns],
            'red_count': self.red_count,
            'blue_count': self.blue_count,
            'red_heading_range': self.red_heading_range,
            'blue_heading_range': self.blue_heading_range,
            'initial_known_enemy_ids': self.initial_known_enemy_ids,
            'ally_policies': {k: v.to_dict() for k, v in self.ally_policies.items()},
            'enemy_policies': {k: v.to_dict() for k, v in self.enemy_policies.items()},
        }


@dataclass
class ActionSpec:
    action_id: str
    description: str
    action_mask: List[int]
    task_slots: TaskSlotSpec
    reward_weights: RewardWeights
    termination: TerminationSpec
    intel: IntelConfig = field(default_factory=IntelConfig)
    reset_templates: List[ResetTemplate] = field(default_factory=list)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ActionSpec':
        return ActionSpec(
            action_id=str(data['action_id']),
            description=str(data.get('description', '')),
            action_mask=[int(x) for x in data.get('action_mask', [0, 1, 2])],
            task_slots=TaskSlotSpec.from_dict(data.get('task_slots', {})),
            reward_weights=RewardWeights.from_dict(data.get('reward_weights', {})),
            intel=IntelConfig.from_dict(data.get('intel', {})),
            termination=TerminationSpec.from_dict(data.get('termination', {})),
            reset_templates=[ResetTemplate.from_dict(x) for x in data.get('reset_templates', [])],
            observation=ObservationConfig.from_dict(data.get('observation', {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'description': self.description,
            'action_mask': self.action_mask,
            'task_slots': self.task_slots.to_dict(),
            'reward_weights': self.reward_weights.to_dict(),
            'intel': self.intel.to_dict(),
            'termination': self.termination.to_dict(),
            'reset_templates': [x.to_dict() for x in self.reset_templates],
            'observation': self.observation.to_dict(),
        }


@dataclass
class SceneConfig:
    scene_id: str
    map_size: List[float]
    obstacles: List[Dict[str, float]]
    reset_templates: List[ResetTemplate]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SceneConfig':
        map_size = [
            float(data.get('map_size', [1000.0, 1000.0])[0]),
            float(data.get('map_size', [1000.0, 1000.0])[1]),
        ]
        obstacles = [_normalize_obstacle(o) for o in data.get('obstacles', [])]
        raw_templates = data.get('reset_templates', data.get('templates', []))
        templates: List[ResetTemplate] = []
        for idx, raw_template in enumerate(raw_templates):
            template_data = dict(raw_template)
            template_data.setdefault('template_id', f'template_{idx}')
            template_data.setdefault('map_size', map_size)
            template_data.setdefault('obstacles', obstacles)
            templates.append(ResetTemplate.from_dict(template_data))
        return SceneConfig(
            scene_id=str(data.get('scene_id', 'default_scene')),
            map_size=map_size,
            obstacles=obstacles,
            reset_templates=templates,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'scene_id': self.scene_id,
            'map_size': self.map_size,
            'obstacles': self.obstacles,
            'reset_templates': [template.to_dict() for template in self.reset_templates],
        }

def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_action_spec(path: str | Path) -> ActionSpec:
    return ActionSpec.from_dict(load_json(path))


def save_action_spec(path: str | Path, spec: ActionSpec) -> None:
    Path(path).write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')


def load_scene_config(path: str | Path) -> SceneConfig:
    return SceneConfig.from_dict(load_json(path))


def save_scene_config(path: str | Path, scene: SceneConfig) -> None:
    Path(path).write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
