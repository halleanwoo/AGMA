import datetime
import os
import pprint
import time
import threading
import torch as th
import random 
import numpy as np
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot



def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    print("=======")
    print(args.device)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    print("\n=================")
    levin_auto_describe = 'env_' + args.env_args['map_name'] + '_' + args.name.split('_')[0]  + '_'  # env map algo
    if args.name.split('_')[0] == 'smix':
        levin_auto_describe = levin_auto_describe + args.mixer + '_'
    # average and sub-average
    # --- convention ----
    if args.t_max==20050000:
        levin_auto_describe = levin_auto_describe + 'long_'

    # -----  QMIX -------
    if args.learner == 'q_learner':
        levin_auto_describe = levin_auto_describe + 'QMIX'
    # -----  convention -----
    elif args.learner == 'q_learner_convention':
        levin_auto_describe = levin_auto_describe + 'Convention'

        if args.flag_latent_expand_layer:
            levin_auto_describe = levin_auto_describe + '_latentExpandDim'
        else:
            levin_auto_describe = levin_auto_describe + '_latentKeepDim'

        if args.flag_gen_loss_norm:
            levin_auto_describe = levin_auto_describe + '_GenLossNorm'
        else:
            levin_auto_describe = levin_auto_describe + '_GenLossNoNorm'

        if args.flag_gen_loss_clamp:
            levin_auto_describe = levin_auto_describe + '_GenLossClamp'
        else:
            levin_auto_describe = levin_auto_describe + '_GenLossNoClamp'

        if args.flag_gen_entropy > 0:
            levin_auto_describe = levin_auto_describe + '_PosEnt'            
        elif args.flag_gen_entropy < 0:
            levin_auto_describe = levin_auto_describe + '_NegEnt' 

        if args.flag_input_only_latent:
            levin_auto_describe = levin_auto_describe + '_InputLatent'         
        elif args.flag_input_only_inputs:
            levin_auto_describe = levin_auto_describe + '_InputInputs'  # 如果没有这一项的填写，则inputs是 input+latent的cat

        if args.flag_pair_gen_loss:
            levin_auto_describe = levin_auto_describe + '_PairGenLossAll'
        elif args.flag_all_gen_loss:
            levin_auto_describe = levin_auto_describe + '_AllGenLoss'

        if args.flag_hyperNet4input:
            levin_auto_describe = levin_auto_describe + '_Hyper4InputDim' + str(args.dim_hyperNet4input) 
        if args.flag_hyperNet4fc1:
            levin_auto_describe = levin_auto_describe + '_Hyper4Fc1Dim' + str(args.dim_hyperNet4input)

        if args.flag_normHyper4input:
            levin_auto_describe = levin_auto_describe + 'Norm'
        else:
            levin_auto_describe = levin_auto_describe + 'noNorm'

    # -----  peception -----
    elif args.learner == 'q_learner_perception':
        levin_auto_describe = levin_auto_describe + 'Perception'

        if args.flag_input_only_latent:
            levin_auto_describe = levin_auto_describe + '_InputLatent'         
        elif args.flag_input_only_inputs:
            levin_auto_describe = levin_auto_describe + '_InputInputs'  # 如果没有这一项的填写，则inputs是 input+latent的cat

        if args.flag_hyperNet4input:
            levin_auto_describe = levin_auto_describe + '_Hyper4InputDim' + str(args.dim_hyperNet4input) 
        if args.flag_hyperNet4fc1:
            levin_auto_describe = levin_auto_describe + '_Hyper4Fc1Dim' + str(args.dim_hyperNet4input)
        if args.flag_hyperNet4fc2:
            levin_auto_describe = levin_auto_describe + '_Hyper4Fc2Dim' + str(args.dim_hyperNet4input)


        if args.flag_normHyper4input:
            levin_auto_describe = levin_auto_describe + 'Norm'
        else:
            levin_auto_describe = levin_auto_describe + 'noNorm'

        # if args.type_fc2 == 'fc':
        #     levin_auto_describe = levin_auto_describe + '_fc2'
        # elif args.type_fc2 == 'hyper_net':
        #     levin_auto_describe = levin_auto_describe + '_hyperNet2'

        if args.flag_latent_expand_layer:
            levin_auto_describe = levin_auto_describe + '_latentExpandDim' + str(args.dim_expand_latent_dim)
        else:
            levin_auto_describe = levin_auto_describe + '_latentKeepDim'
        
        if not args.flag_latent_state_loss:
            levin_auto_describe = levin_auto_describe + '_noStateLoss'
        else:
            if args.flag_latent_state_loss > 0:
                levin_auto_describe = levin_auto_describe + '_StateLoss'
            else:
                levin_auto_describe = levin_auto_describe + '_NegStateLoss'
            if (args.loss_entropy_weight != 1) or (args.loss_kl_weight != 1):
                levin_auto_describe = levin_auto_describe + 'Ent' + str(args.loss_entropy_weight).replace('.','d') + 'KL' + str(args.loss_kl_weight).replace('.','d')

        levin_auto_describe = levin_auto_describe + '_VarClamp' + str(args.var_floor).replace('.','d')   # 之前没写的都是 0.02        
        levin_auto_describe = levin_auto_describe + '_latentDim' + str(args.latent_dim)

        # --- consesus
        if args.flag_consensus_loss_pair:
            levin_auto_describe = levin_auto_describe + '_consensusLossPair' + str(args.consensus_loss_weight).replace('.','d')
        elif args.flag_consensus_loss_all:
            levin_auto_describe = levin_auto_describe + '_consensusLossAll' + str(args.consensus_loss_weight).replace('.','d')
        else:
            pass
        if args.flag_consensus_after_fcDesign and args.flag_latent_expand_layer:
            levin_auto_describe = levin_auto_describe + 'MSE4fc'

    if args.use_cuda:
        levin_auto_describe = levin_auto_describe + '_gpu'
    levin_auto_describe = levin_auto_describe + '_' + str(args.seed)

    # # ----- Sub-AVG -----
    # if args.levin_flag_average_dqn==0 and args.levin_flag_mixer_average==0:
    #     pass
    # else:
    #     if args.levin_flag_average_dqn==1:
    #         levin_auto_describe = levin_auto_describe + 'average' + str(args.average_N_target)
    #         if args.levin_flag_average_dqn_select < 0:
    #             levin_auto_describe = levin_auto_describe + 'neg' + ("Mean" if args.levin_name_select_replacement=='mean' else "Zero_start2") + '_'
    #         elif args.levin_flag_average_dqn_select > 0:
    #             levin_auto_describe = levin_auto_describe + 'pos' + ("Mean" if args.levin_name_select_replacement=='mean' else "Zero_start2") + '_'
    #     if args.levin_flag_mixer_average==1:
    #         levin_auto_describe = levin_auto_describe + 'mix' + str(args.average_N_mixer_target)
    #         if args.levin_flag_mixer_select < 0:
    #             levin_auto_describe = levin_auto_describe + 'neg' + ("Mean" if args.levin_name_mixer_select_replacement=='mean' else "Zero_start2") + '_'
    #         elif args.levin_flag_mixer_select > 0:
    #             levin_auto_describe = levin_auto_describe + 'pos' + ("Mean" if args.levin_name_mixer_select_replacement=='mean' else "Zero_start2") + '_'
    # if args.levin_flag_average_N_min:
    #     levin_auto_describe = levin_auto_describe + 'average_N_min' + '_'
    # if args.levin_flag_mix_N_min:
    #     levin_auto_describe = levin_auto_describe + 'mix_N_min' + '_'
    # if args.levin_flag_mix_mean_minus_var:
    #     levin_auto_describe = levin_auto_describe + 'MeanMinusVar' + '_'

    # # double
    # if args.double_q:
    #     if args.levin_flag_beta_double_q:
    #         levin_auto_describe = levin_auto_describe + 'betaDouble' + str(args.levin_double_q_beta)
    #     else:
    #         levin_auto_describe = levin_auto_describe + 'Double'
    # else:
    #     levin_auto_describe = levin_auto_describe + 'noDouble'
    # if args.target_update_interval != 200:
    #     levin_auto_describe = levin_auto_describe + str(args.target_update_interval)

    # if not args.double_q:
    #     levin_auto_describe = levin_auto_describe + 'noDouble'
    # else:
    #     if not args.levin_flag_beta_double_q:
    #         levin_auto_describe = levin_auto_describe + 'Double'
    #     else:
    #         levin_auto_describe = levin_auto_describe + 'betaDouble' + str(args.levin_double_q_beta)
    args.levin_z_auto_describe = levin_auto_describe
    print(levin_auto_describe)

    print(args.levin_z_describe)

    print("using average DQN:", end=' ')
    print("Yes  target: %d" % args.average_N_target if args.levin_flag_average_dqn else "No", end=' | ')
    print("select > or < average:", end=' ')
    print("Yes  signal: %d" % args.levin_flag_average_dqn_select if args.levin_flag_average_dqn_select else "No", end=' | ')
    print("replace average_select by :", end=' ')
    print("mean" if args.levin_name_select_replacement=='mean' else "zero")

    print("using average mixer:", end=' ')
    print("Yes  mixer: %d" % args.average_N_mixer_target if args.levin_flag_mixer_average else "No", end=' | ')
    print("mixer: select > or <:", end=' ')
    print("Yes  signal: %d" % args.levin_flag_mixer_select if args.levin_flag_mixer_select else "No", end=' | ')
    print("replace mixer_select by :", end=' ')
    print("mean" if args.levin_name_mixer_select_replacement=='mean' else "zero")

    print("using double_q: ", end=' ')
    print('Yes' if args.double_q else "No", end=' | ')
    print("using double_q with beta_prob ", end=' ')
    print('Yes  beta: %f'%(args.levin_double_q_beta) if args.levin_flag_beta_double_q else "No")

    print("\n-----------------")
    print("using lambda average DQN:", end=' ')
    print("Yes  lambda: %f" % args.levin_average_lambda if args.levin_flag_lambda_average else "No")

    # -----------------------------
    print("==================\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)



    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    # import ipdb
    # ipdb.set_trace()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        if runner.t_env > args.t_max - 10000:
            args.flag_log_latentFinal = 1

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        # import ipdb
        # ipdb.set_trace()
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
