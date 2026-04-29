import numpy as np
import torch

from ppo.registry_io import load_policy_bundle

BUNDLE_DIR = 'actions/trained/move_easy/last'
EPISODES = 200
SEED = 17
DEVICE = 'cpu'
RENDER = True
DETERMINISTIC = False


if __name__ == '__main__':
    policy, action_spec, _ = load_policy_bundle(BUNDLE_DIR, device=DEVICE)
    from env.combat_env import SingleSkillEnv

    env = SingleSkillEnv(action_spec, seed=SEED, headless=not RENDER, controlled_red_id=0)
    success_flags = []
    episode_steps = []
    episode_returns = []

    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        step = 0
        ep_return = 0.0
        info = {}
        while not done and step < action_spec.termination.max_steps:
            obs_t = {k: torch.as_tensor(v[None, ...], dtype=torch.float32, device=DEVICE) for k, v in obs.items()}
            with torch.no_grad():
                act, _, _, _ = policy.act(obs_t, action_spec.action_mask, deterministic=DETERMINISTIC)
            obs, reward, done, info = env.step(act.detach().cpu().numpy()[0].tolist())
            if RENDER:
                env.base_env.render()
            ep_return += float(reward)
            step += 1

        success = bool(info.get('success', False))
        success_flags.append(1.0 if success else 0.0)
        episode_steps.append(step)
        episode_returns.append(ep_return)

        if (ep + 1) % 20 == 0:
            print(
                f'ep={ep + 1:04d} '
                f'success_rate={np.mean(success_flags):.3f} '
                f'avg_steps={np.mean(episode_steps):.1f} '
                f'avg_return={np.mean(episode_returns):.3f}'
            )

    print('--- final ---')
    print(f'success_rate: {np.mean(success_flags):.4f}')
    print(f'avg_steps: {np.mean(episode_steps):.2f}')
    print(f'avg_return: {np.mean(episode_returns):.4f}')

