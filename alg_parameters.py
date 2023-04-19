import datetime

""" Hyperparameters"""


class EnvParameters:
    LOCAL_N_AGENTS = 8  # number of agents used in training
    GLOBAL_N_AGENT=(400,600)  # 300: pp CAN NOT SOLVE (300,500)
    N_ACTIONS = 5
    EPISODE_LEN = 356  # maximum episode length in training
    FOV_SIZE = 9
    WORLD_SIZE = (40, 60)
    OBSTACLE_PROB = (0.0, 0.3)
    ACTION_COST = -0.3
    IDLE_COST = -0.3
    GOAL_REWARD = 0.0
    COLLISION_COST = -2
    BLOCKING_COST = -1


class TrainingParameters:
    lr = 1e-5
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # For GAE
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 10
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.08
    POLICY_COEF = 10
    VALID_COEF = 0.5
    BLOCK_COEF = 0.5
    N_EPOCHS = 10
    N_ENVS = 16  # number of processes
    N_MAX_STEPS = 5e7  # maximum number of time steps used in training
    N_STEPS = 2 ** 10  # number of time steps per process per data collection
    MINIBATCH_SIZE = int(2 ** 10)
    Destroy_factor=0.05
    ITERATION_LIMIT=80


class NetParameters:
    NET_SIZE = 512
    NUM_CHANNEL = 10  # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 4 # [dx, dy, d total, action t-1]


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1


class RecordingParameters:
    RETRAIN = False
    WANDB =  True
    ENTITY = 'yutong'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'MAPF'
    EXPERIMENT_NAME = 'LNS2+RL_allow_collision'
    EXPERIMENT_NOTE = ''
    SAVE_INTERVAL = 2e6  # interval of saving model
    GLOBAL_INTERVAL=5
    GRAD_LOGFREQ=1000
    EVAL=False
    BEST_INTERVAL = 0  # interval of saving model with the best performance
    GIF_INTERVAL = 1e6  # interval of saving gif
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS  # interval of evaluating training model0
    EVAL_EPISODES = 1  # number of episode used in evaluation
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    LOSS_NAME = ['all_loss', 'policy_loss', 'policy_entropy', 'critic_loss', 'valid_loss',
                 'blocking_loss', 'clipfrac',
                 'grad_norm', 'advantage']


all_args = {'LOCAL_N_AGENTS': EnvParameters.LOCAL_N_AGENTS, "GLOBAL_N_AGENT":EnvParameters.GLOBAL_N_AGENT,
            'N_ACTIONS': EnvParameters.N_ACTIONS,
            'EPISODE_LEN': EnvParameters.EPISODE_LEN, 'FOV_SIZE': EnvParameters.FOV_SIZE,
            'WORLD_SIZE': EnvParameters.WORLD_SIZE,
            'OBSTACLE_PROB': EnvParameters.OBSTACLE_PROB,
            'ACTION_COST': EnvParameters.ACTION_COST,
            'IDLE_COST': EnvParameters.IDLE_COST, 'GOAL_REWARD': EnvParameters.GOAL_REWARD,
            'COLLISION_COST': EnvParameters.COLLISION_COST,
            'BLOCKING_COST': EnvParameters.BLOCKING_COST,
            'lr': TrainingParameters.lr, 'GAMMA': TrainingParameters.GAMMA, 'LAM': TrainingParameters.LAM,
            'CLIPRANGE': TrainingParameters.CLIP_RANGE, 'MAX_GRAD_NORM': TrainingParameters.MAX_GRAD_NORM,
            'ENTROPY_COEF': TrainingParameters.ENTROPY_COEF,
            'VALUE_COEF': TrainingParameters.VALUE_COEF,
            'POLICY_COEF': TrainingParameters.POLICY_COEF,
            'VALID_COEF': TrainingParameters.VALID_COEF, 'BLOCK_COEF': TrainingParameters.BLOCK_COEF,
            'N_EPOCHS': TrainingParameters.N_EPOCHS, 'N_ENVS': TrainingParameters.N_ENVS,
            'N_MAX_STEPS': TrainingParameters.N_MAX_STEPS,
            'N_STEPS': TrainingParameters.N_STEPS, 'MINIBATCH_SIZE': TrainingParameters.MINIBATCH_SIZE,
            "Destroy_factor":TrainingParameters.Destroy_factor,
            'NET_SIZE': NetParameters.NET_SIZE, 'NUM_CHANNEL': NetParameters.NUM_CHANNEL,
            'GOAL_REPR_SIZE': NetParameters.GOAL_REPR_SIZE, 'VECTOR_LEN': NetParameters.VECTOR_LEN,
            'SEED': SetupParameters.SEED, 'USE_GPU_LOCAL': SetupParameters.USE_GPU_LOCAL,
            'USE_GPU_GLOBAL': SetupParameters.USE_GPU_GLOBAL,
            'NUM_GPU': SetupParameters.NUM_GPU, 'RETRAIN': RecordingParameters.RETRAIN,
            'WANDB': RecordingParameters.WANDB,
            'ENTITY': RecordingParameters.ENTITY,
            'TIME': RecordingParameters.TIME, 'EXPERIMENT_PROJECT': RecordingParameters.EXPERIMENT_PROJECT,
            'EXPERIMENT_NAME': RecordingParameters.EXPERIMENT_NAME,
            'EXPERIMENT_NOTE': RecordingParameters.EXPERIMENT_NOTE,
            'SAVE_INTERVAL': RecordingParameters.SAVE_INTERVAL,"EVAL":RecordingParameters.EVAL, "BEST_INTERVAL": RecordingParameters.BEST_INTERVAL,
            'GIF_INTERVAL': RecordingParameters.GIF_INTERVAL, 'EVAL_INTERVAL': RecordingParameters.EVAL_INTERVAL,
            'EVAL_EPISODES': RecordingParameters.EVAL_EPISODES, 'RECORD_BEST': RecordingParameters.RECORD_BEST,
            'MODEL_PATH': RecordingParameters.MODEL_PATH, 'GIFS_PATH': RecordingParameters.GIFS_PATH}

