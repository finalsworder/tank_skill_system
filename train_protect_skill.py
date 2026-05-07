from train_skill import TrainRuntimeConfig, train_from_config_paths

TASK_CONFIG = 'configs/tasks/protect_task.json'
SCENE_CONFIG = 'configs/scenes/protect_scene.json'
OUTPUT_DIR = 'actions/trained/protect'
NUM_ENVS = 32
EPISODES_PER_ENV = 500
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
LEARNING_RATE = 3e-4
DEVICE = 'cuda:0'
SEED = 7
CONTROLLED_RED_ID = 0
CHECKPOINT_INTERVAL = 25
MAX_TIME = 300
PROGRESS_BAR = True
USE_SWANLAB = True
SWANLAB_PROJECT = 'tank-protect-skill'
POLICY_HIDDEN_DIM = 256
POLICY_ATTENTION_HEADS = 4
POLICY_SELF_LAYERS = 3
POLICY_ENTITY_LAYERS = 2
POLICY_TRUNK_LAYERS = 3
POLICY_ACTIVATION = 'silu'

if __name__ == '__main__':
    train_from_config_paths(
        TASK_CONFIG,
        SCENE_CONFIG,
        TrainRuntimeConfig(
            output_dir=OUTPUT_DIR,
            num_envs=NUM_ENVS,
            episodes_per_env=EPISODES_PER_ENV,
            update_epochs=UPDATE_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            learning_rate=LEARNING_RATE,
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
        ),
    )
