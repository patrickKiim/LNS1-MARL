import datetime

""" Hyperparameters"""


class EnvParameters:  # (0.00625,0.125),( 0.01234,0.176),(0.02,0.26),(0.03,0.39),(0.069,0.48),(0.11,0.675)
    LOCAL_N_AGENTS_LIST = [4,6,8,8,8,8]  # number of agents used in training
    GLOBAL_N_AGENT_LIST=[30,60,100,200,300,400]  # 300: pp CAN NOT SOLVE (300,500)
    N_ACTIONS = 5
    EPISODE_LEN = 356  # maximum episode length in training
    FOV_SIZE = 9
    WORLD_SIZE_LIST = [15,20,25,30,35,40] # 0.1,0.16,0.225,0.298,0.36,0.415
    OBSTACLE_PROB_LIST = [0.02,0.04,0.07,0.1,0.15,0.2]
    ACTION_COST = -0.3
    IDLE_COST = -0.3
    MOVE_BACK_COST=-0.4
    GOAL_REWARD = 0.0
    DY_COLLISION_COST = -1
    AG_COLLISION_COST = -1.7
    BLOCKING_COST = -1
    NUM_TIME_SLICE=6
    WINDOWS=15
    OFF_ROUTE_FACTOR=0.05
    DIS_FACTOR = 0.2
    SWEITCH_TIMESTEP=[1e6,2e6,3e6,6e6,1e7]
    INFLATION=10


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
    N_MAX_STEPS = 6e7  # maximum number of time steps used in training
    N_STEPS = 2 ** 10  # number of time steps per process per data collection
    MINIBATCH_SIZE =int(2**10)#int(2 ** 10) #int(2 ** 10)
    Destroy_factor=0.05
    ITERATION_LIMIT_LIST=[5,20,50,600,800,900]
    opti_eps=1e-5
    huber_delta=10
    weight_decay=0


class NetParameters:
    NET_SIZE = 512
    NUM_CHANNEL = 15  # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 9 # [dx, dy, d,self collsion ratio, total collsion ration,step vs episode length, step vs makespan, ratio of current reached goal, old action]
    GAIN=0.01


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1


class RecordingParameters:
    RETRAIN = False
    WANDB = True
    ENTITY = 'yutong'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'MAPF'
    EXPERIMENT_NAME = 'sipps_block_individual_if_better_moreinfo'
    EXPERIMENT_NOTE = 'timeslice, allow_collision, collision penalty=1,1.7, remove back action from mask and train it by penalty=-0.4,' \
                      'add off-rout rewards based on sipps+pp, time_windows=15, added corresponding SIPPS as input of the whole net, add reward shaping, modified reward structure,' \
                      'express the number of ' \
                      'dynamic obstacles and agents in the obs, penalty*num_collision, use new training framework, add tricks in mappo, curriculum learning based on time step,' \
                      'give distance penalty only when individual policy fail, add more global info eg:step vs episode length, ratio of current reached goal'
    SAVE_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS*100  # interval of saving model
    GLOBAL_INTERVAL=5
    GRAD_LOGFREQ=1000
    EVAL=False
    BEST_INTERVAL = 0  # interval of saving model with the best performance
    GIF_INTERVAL = 1e6  # interval of saving gif
    EVAL_INTERVAL_SCALA = SAVE_INTERVAL
    EVAL_INTERVAL_GLOBAL = SAVE_INTERVAL*4  # interval of evaluating training model0
    EVAL_EPISODES = 1  # number of episode used in evaluation
    EVAL_MAX_ITERATION=400
    EVAL_NUM_AGENT_GLOBAL = 8
    EVAL_NUM_AGENT=64
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    LOSS_NAME = ['all_loss', 'policy_loss', 'policy_entropy', 'critic_loss', 'valid_loss',
                 'blocking_loss', 'clipfrac',
                 'grad_norm', 'advantage',"prop_policy","prop_en","prop_v","prop_valid","prop_block"]

all_args = {'LOCAL_N_AGENTS': EnvParameters.LOCAL_N_AGENTS_LIST, "GLOBAL_N_AGENT":EnvParameters.GLOBAL_N_AGENT_LIST,
            'N_ACTIONS': EnvParameters.N_ACTIONS,'DIS_FACTOR':EnvParameters.DIS_FACTOR,
            'EPISODE_LEN': EnvParameters.EPISODE_LEN, 'FOV_SIZE': EnvParameters.FOV_SIZE,
            'WORLD_SIZE': EnvParameters.WORLD_SIZE_LIST,
            'OBSTACLE_PROB': EnvParameters.OBSTACLE_PROB_LIST,
            'ACTION_COST': EnvParameters.ACTION_COST,
            'IDLE_COST': EnvParameters.IDLE_COST, 'GOAL_REWARD': EnvParameters.GOAL_REWARD,
            'AG_COLLISION_COST': EnvParameters.AG_COLLISION_COST,
            'DY_COLLISION_COST': EnvParameters.DY_COLLISION_COST,
            'BLOCKING_COST': EnvParameters.BLOCKING_COST,'NUM_TIME_SLICE':EnvParameters.NUM_TIME_SLICE,
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
            'GIF_INTERVAL': RecordingParameters.GIF_INTERVAL, 'EVAL_INTERVAL_SCALA': RecordingParameters.EVAL_INTERVAL_SCALA,'EVAL_INTERVAL_GLOBAL': RecordingParameters.EVAL_INTERVAL_GLOBAL,
            'EVAL_EPISODES': RecordingParameters.EVAL_EPISODES, 'RECORD_BEST': RecordingParameters.RECORD_BEST,
            'MODEL_PATH': RecordingParameters.MODEL_PATH, 'GIFS_PATH': RecordingParameters.GIFS_PATH}

