from __future__ import annotations

def compute_weighted_reward(events, weights):
    return float(sum(float(weights[k]) * float(events.get(k,0.0)) for k in weights))
