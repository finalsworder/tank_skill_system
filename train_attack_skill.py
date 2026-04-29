from train_skill import TrainRuntimeConfig, train_from_spec_paths

ACTION_SPEC = 'actions/specs/attack_action.json'
SCENE_CONFIG = 'configs/scenes/attack_scene.json'
OUTPUT_DIR = 'actions/trained/attack'
NUM_ENVS = 100
EPISODES_PER_ENV = 1000
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 1024
LEARNING_RATE = 3e-4
ENT_COEF = 0.003
ENT_COEF_FINAL = 0.0003
DEVICE = 'cuda:0'
SEED = 2
CONTROLLED_RED_ID = 0
CHECKPOINT_INTERVAL = 50
MAX_TIME = 1000
PROGRESS_BAR = True
USE_SWANLAB = True
SWANLAB_PROJECT = 'tank-attack-skill'
POLICY_HIDDEN_DIM = 256
POLICY_ATTENTION_HEADS = 4
POLICY_SELF_LAYERS = 3
POLICY_ENTITY_LAYERS = 2
POLICY_TRUNK_LAYERS = 3
POLICY_ACTIVATION = 'silu'
POLICY_LOG_STD_INIT = -0.5
POLICY_LOG_STD_MIN = -3.0
POLICY_LOG_STD_MAX = -0.5

if USE_SWANLAB:
    import swanlab
    swanlab.login(api_key="CboKBhbXjjOrAjg5ACIV0", save=True)

if __name__ == '__main__':
    train_from_spec_paths(
        ACTION_SPEC,
        SCENE_CONFIG,
        TrainRuntimeConfig(
            output_dir=OUTPUT_DIR,
            num_envs=NUM_ENVS,
            episodes_per_env=EPISODES_PER_ENV,
            update_epochs=UPDATE_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            learning_rate=LEARNING_RATE,
            ent_coef=ENT_COEF,
            ent_coef_final=ENT_COEF_FINAL,
            device=DEVICE,
            seed=SEED,
            controlled_red_id=CONTROLLED_RED_ID,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            max_time=MAX_TIME,
            progress_bar=PROGRESS_BAR,
            use_swanlab=USE_SWANLAB,
            swanlab_project=SWANLAB_PROJECT,
            policy_hidden_dim=POLICY_HIDDEN_DIM,
            policy_attention_heads=POLICY_ATTENTION_HEADS,
            policy_self_layers=POLICY_SELF_LAYERS,
            policy_entity_layers=POLICY_ENTITY_LAYERS,
            policy_trunk_layers=POLICY_TRUNK_LAYERS,
            policy_activation=POLICY_ACTIVATION,
            policy_log_std_init=POLICY_LOG_STD_INIT,
            policy_log_std_min=POLICY_LOG_STD_MIN,
            policy_log_std_max=POLICY_LOG_STD_MAX,
        ),
    )
