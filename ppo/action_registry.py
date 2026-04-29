from __future__ import annotations
from pathlib import Path
class ActionRegistry:
    def __init__(self, root_dir): self.root_dir=Path(root_dir)
    def list_actions(self): return sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
    def bundle_dir(self, action_id): return str(self.root_dir/action_id/'best')
