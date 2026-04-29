import torch
from ppo.registry_io import load_policy_bundle

BUNDLE_DIR = 'actions/trained/attack/20260424154912/episode_0400'
EPISODES = 10000
SEED = 12
DEVICE = 'cpu'
if __name__ == '__main__':
    POLICY, ACTION_SPEC, OBS_META = load_policy_bundle(BUNDLE_DIR, device=DEVICE)
    from env.combat_env import SingleSkillEnv

    ENV = SingleSkillEnv(ACTION_SPEC, seed=SEED, headless=False, controlled_red_id=0)
    for EP in range(EPISODES):
        OBS = ENV.reset()
        DONE = False
        STEP = 0
        while not DONE and STEP < ACTION_SPEC.termination.max_steps:
            OBS_T = {k: torch.as_tensor(v[None, ...], dtype=torch.float32, device=DEVICE) for k, v in OBS.items()}
            with torch.no_grad():
                ACT, _, _, _ = POLICY.act(OBS_T, ACTION_SPEC.action_mask, deterministic=True)
            OBS, REWARD, DONE, INFO = ENV.step(ACT.detach().cpu().numpy()[0].tolist())
            ENV.base_env.render()
            STEP += 1
        print('episode', EP, 'done', DONE, 'info', INFO)
