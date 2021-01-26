import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)

    # --- levin设置 random seed ---
    if config['levin_set_seed'] != 0:
        config["seed"] = config['levin_set_seed']

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)


    # ===== levin confit =====
    # -------  distributional ------
    config_dict['levin_flag_quantile'] = 0
    config_dict['N_QUANT'] = 50
    config_dict['levin_flag_shangtong_loss'] = 0
    config_dict['mixing_embed_dim'] = 16

    # ---------   average target -------
    config_dict['double_q'] = 1
    # 不互斥
    config_dict['levin_flag_beta_double_q'] = 0
    config_dict['levin_double_q_beta'] = 0.5     # 小于 beta时，使用 double Q

    config_dict['levin_flag_average_dqn'] = 0
    config_dict['average_N_target'] = 5
    # note: 要使用，需要保证 levin_flag_average_dqn = 1
    # 使用N个target在每个状态上的最小值去求target，而不是使用平均值
    config_dict['levin_flag_average_N_min'] = 0 # 保证levin_flag_average_dqn=1，此flag为1时，则不使用下面的select，而是直接选择N个中最小的
    # 选择是使用大于平均值的，还是小于平均值的
    config_dict['levin_flag_average_dqn_select'] = 0   # 当为1时，取大的；当为-1时，取小的；当为0时，直接使用平均值
    # 确定替换之后，选择使用平均值mean来替换，还是zero来替换
    config_dict['levin_name_select_replacement'] = 'mean'
    
    # note: 要使用，需要保证 levin_flag_average_dqn = 1
    config_dict['levin_flag_lambda_average'] = 0           # 不是直接除以 average_N_target， 而是使用 td-lambda的形式进行滑动平均
    config_dict['levin_average_lambda'] = 0.95

    config_dict['levin_flag_mixer_average'] = 0
    config_dict['average_N_mixer_target'] = 5
    config_dict['levin_flag_mixer_select'] = 0   # 当为1时，取大的；当为-1时，取小的
    config_dict['levin_name_mixer_select_replacement'] = 'mean'  # 当为1时，取大的；当为-1时，取小的
    config_dict['levin_flag_mix_N_min'] = 0 # levin_flag_mixer_average=1，此flag为1时，则不使用上面的select，而是直接选择N个中最小的
    config_dict['levin_flag_mix_mean_minus_var'] = 0 # 使用条件:levin_flag_mixer_average=1，此flag为1时，则不使用上面的select，而是直接选择 mean - std

    
    config_dict['levin_flag_average_not_averaging'] = 0     # 将所有deepcopy的target拿去计算td-error，而不是使用平均值（未实现）

    config_dict['levin_z_auto_describe'] = 'xxxx'
    config_dict['levin_z_describe'] = 'xxxx'
    # ================================

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # print(config_dict.keys())
    # import ipdb
    # ipdb.set_trace()

    ex.run_commandline(params)


