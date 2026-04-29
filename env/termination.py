from __future__ import annotations

def evaluate_termination(events, spec, step_count):
    success=any(bool(events.get(name,False)) for name in spec.success)
    failure=any(bool(events.get(name,False)) for name in spec.failure)
    truncated=step_count >= spec.max_steps
    return success or failure or truncated, success, {'success':success,'failure':failure,'truncated':truncated}
