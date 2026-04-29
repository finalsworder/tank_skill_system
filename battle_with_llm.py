from ppo.action_registry import ActionRegistry
from ppo.actor_pool import ActorPool
PLANNING_INTERVAL=100
DEVICE='cpu'
ACTIONS_DIR='actions/trained'
if __name__=='__main__':
    print('请在接入真实API和完整runtime后使用该脚本。当前工程已提供ActorPool与LLM接口。')
