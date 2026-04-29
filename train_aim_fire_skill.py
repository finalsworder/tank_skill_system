from train_skill import TrainRuntimeConfig, train_from_spec_paths

ACTION_SPEC = 'actions/specs/aim_fire_action.json'
SCENE_CONFIG = 'configs/scenes/aim_fire_scene.json'
OUTPUT_DIR = 'actions/trained/aim_fire'
NUM_ENVS = 32
EPISODES_PER_ENV = 300
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 512
LEARNING_RATE = 3e-4
ENT_COEF = 0.003
ENT_COEF_FINAL = 0.0003
DEVICE = 'cuda:0'
SEED = 3
CONTROLLED_RED_ID = 0
CHECKPOINT_INTERVAL = 25
MAX_TIME = 300
PROGRESS_BAR = True
USE_SWANLAB = True
SWANLAB_PROJECT = 'tank-aim-fire-skill'
POLICY_HIDDEN_DIM = 128
POLICY_ATTENTION_HEADS = 2
POLICY_SELF_LAYERS = 2
POLICY_ENTITY_LAYERS = 1
POLICY_TRUNK_LAYERS = 2
POLICY_ACTIVATION = 'tanh'
POLICY_LOG_STD_INIT = -0.5
POLICY_LOG_STD_MIN = -3.0
POLICY_LOG_STD_MAX = -0.5

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
