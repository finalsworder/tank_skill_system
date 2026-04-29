from train_skill import TrainRuntimeConfig, train_from_spec_paths

ACTION_SPEC = 'actions/specs/move_easy_action.json'
SCENE_CONFIG = 'configs/scenes/move_easy_scene.json'
OUTPUT_DIR = 'actions/trained/move_easy'
NUM_ENVS = 16
EPISODES_PER_ENV = 1000
UPDATE_EPOCHS = 2
MINIBATCH_SIZE = 128
LEARNING_RATE = 3e-4
ENT_COEF = 0.003
ENT_COEF_FINAL = 0.0003
DEVICE = 'cuda:0'
SEED = 7
CONTROLLED_RED_ID = 0
CHECKPOINT_INTERVAL = 10
MAX_TIME = 200
PROGRESS_BAR = True
USE_SWANLAB = True
SWANLAB_PROJECT = 'tank-move-easy-skill'
POLICY_HIDDEN_DIM = 256
POLICY_ATTENTION_HEADS = 2
POLICY_SELF_LAYERS = 3
POLICY_SELF_ENCODER = 'raw'
POLICY_ENTITY_LAYERS = 1
POLICY_TRUNK_LAYERS = 2
POLICY_ACTIVATION = 'silu'
POLICY_LOG_STD_INIT = -0.5
POLICY_LOG_STD_MIN = -3.0
POLICY_LOG_STD_MAX = -0.5
RADAR_CNN_CHANNELS1 = 16
RADAR_CNN_CHANNELS2 = 32
RADAR_CNN_KERNEL1 = 5
RADAR_CNN_KERNEL2 = 3
RADAR_CNN_POOL = 'max'

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
            policy_self_encoder=POLICY_SELF_ENCODER,
            policy_entity_layers=POLICY_ENTITY_LAYERS,
            policy_trunk_layers=POLICY_TRUNK_LAYERS,
            policy_activation=POLICY_ACTIVATION,
            policy_log_std_init=POLICY_LOG_STD_INIT,
            policy_log_std_min=POLICY_LOG_STD_MIN,
            policy_log_std_max=POLICY_LOG_STD_MAX,
            radar_cnn_channels1=RADAR_CNN_CHANNELS1,
            radar_cnn_channels2=RADAR_CNN_CHANNELS2,
            radar_cnn_kernel1=RADAR_CNN_KERNEL1,
            radar_cnn_kernel2=RADAR_CNN_KERNEL2,
            radar_cnn_pool=RADAR_CNN_POOL,
        ),
    )

