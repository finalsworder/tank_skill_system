from __future__ import annotations
class SwanLabLogger:
    def __init__(self, enabled=False, project='tank-skill'):
        self.enabled=enabled; self.project=project; self.backend=None
        if enabled:
            import swanlab
            self.backend=swanlab
            self.backend.init(project=project)
    def log(self, data):
        if self.enabled: self.backend.log(data)
