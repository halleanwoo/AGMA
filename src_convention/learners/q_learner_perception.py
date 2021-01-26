import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer, QMixerQuantilie
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
import numpy as np


# ------- levin ------
flag_levin = 0
def huber(x, k=1.0):
    return th.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k)).cuda()


class QLearnerPerception:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.n_actions_levin = args.n_actions

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                if not self.args.levin_flag_quantile:
                    self.mixer = QMixer(args)
                else:
                    self.mixer = QMixerQuantilie(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            if not args.levin_flag_mixer_average:
                self.target_mixer = copy.deepcopy(self.mixer)
            elif args.mixer == "qmix":
                self.target_mixer_list = []
                for i in range(self.args.average_N_mixer_target):
                    self.target_mixer_list.append(copy.deepcopy(self.mixer))
                self.levin_iter_target_mixer_update = 0

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_levin = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        if not self.args.levin_flag_average_dqn:
            self.target_mac = copy.deepcopy(mac)
        else:
            self.target_mac_list = []
            for i in range(self.args.average_N_target):
                self.target_mac_list.append(copy.deepcopy(mac))
            self.levin_iter_target_update = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        # ====== levin =====
        self.number = 0
        self.args.levin_average_lambda = th.tensor(self.args.levin_average_lambda)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, epsilon_levin=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        latent_loss_list = []
        latent_loss_sum = 0
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):                           
            agent_outs, latent_loss = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            latent_loss_sum = latent_loss_sum + latent_loss
            latent_loss_list.append(latent_loss)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time          
        # latent_loss = th.stack(latent_loss_list, dim=-1)
        latent_loss = latent_loss_sum / len(latent_loss_list)

        if self.args.levin_flag_quantile:
            actions = actions.unsqueeze(4)
            actions = actions.expand(actions.size(0), actions.size(1), actions.size(2), actions.size(3), self.args.N_QUANT)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        if not self.args.levin_flag_average_dqn:
            self.target_mac.init_hidden(batch.batch_size)
        else:
            for i in range(self.args.average_N_target):
                self.target_mac_list[i].init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if not self.args.levin_flag_average_dqn:
                target_agent_outs, latent_loss_target = self.target_mac.forward(batch, t=t)   
            
            else:
                target_agent_outs = 0
                
                if self.args.levin_flag_lambda_average:
                    for i in range(self.args.average_N_target):
                        lambda_pow = (self.number - i + self.args.average_N_target) if self.number < i else (self.number - i)
                        target_agent_outs = target_agent_outs + self.target_mac_list[i].forward(batch, t=t) * self.args.levin_average_lambda.pow(lambda_pow)
                    target_agent_outs = target_agent_outs * (1 - self.args.levin_average_lambda)
                elif self.args.levin_flag_average_N_min:
                    for i in range(self.args.average_N_target):
                        target_agent_out = self.target_mac_list[i].forward(batch, t=t)
                        if i == 0:
                            target_agent_outs = target_agent_out
                        else:
                            target_agent_outs = th.where(target_agent_outs < target_agent_out, target_agent_outs, target_agent_out)
                
                else:
                    self.target_agent_out_list = []
                    for i in range(self.args.average_N_target):
                        target_agent_out = self.target_mac_list[i].forward(batch, t=t)
                        target_agent_outs = target_agent_outs + target_agent_out
                        
                        if self.args.levin_flag_average_dqn_select:
                            self.target_agent_out_list.append(target_agent_out)
                    target_agent_outs = target_agent_outs / self.args.average_N_target
                    
                    if self.args.levin_flag_average_dqn_select:
                        
                        if self.args.levin_name_select_replacement == 'mean':    
                            target_out_select_sum = 0
                            for i in range(self.args.average_N_target):
                                if self.args.levin_flag_average_dqn_select > 0:
                                    target_out_select = th.where(self.target_agent_out_list[i] < target_agent_outs, target_agent_outs, self.target_agent_out_list[i])
                                else:
                                    target_out_select = th.where(self.target_agent_out_list[i] > target_agent_outs, target_agent_outs, self.target_agent_out_list[i])
                                target_out_select_sum = target_out_select_sum + target_out_select
                            target_agent_outs = target_out_select_sum / self.args.average_N_target
                        elif self.args.levin_name_select_replacement == 'zero':    
                            target_out_select_sum = 0
                            target_select_bool_sum = 0
                            for i in range(self.args.average_N_target):
                                if self.args.levin_flag_average_dqn_select > 0:
                                    target_select_bool = (self.target_agent_out_list[i] > target_agent_outs).float()
                                    target_out_select = th.where(self.target_agent_out_list[i] > target_agent_outs, self.target_agent_out_list[i], th.full_like(target_agent_outs,0))
                                else:
                                    target_select_bool = (self.target_agent_out_list[i] < target_agent_outs).float()
                                    target_out_select = th.where(self.target_agent_out_list[i] < target_agent_outs, self.target_agent_out_list[i], th.full_like(target_agent_outs,0))
                                target_select_bool_sum = target_select_bool_sum + target_select_bool
                                target_out_select_sum = target_out_select_sum + target_out_select
                            
                            if self.levin_iter_target_update < 2: 
                                pass # print("using average directly")
                            else:
                                target_agent_outs = target_out_select_sum / target_select_bool_sum
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time   

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            if not self.args.levin_flag_beta_double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]

                if self.args.levin_flag_quantile:
                    cur_max_actions = mac_out_detach[:, 1:].mean(dim=4).max(dim=3, keepdim=True)[1]
                    
                    cur_max_actions = cur_max_actions.unsqueeze(4)
                    cur_max_actions = cur_max_actions.expand(cur_max_actions.size(0), cur_max_actions.size(1), cur_max_actions.size(2), cur_max_actions.size(3), self.args.N_QUANT)
                    
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            else:
                if np.random.rand() < self.args.levin_double_q_beta:
                    # Get actions that maximise live Q (for double q-learning)
                    mac_out_detach = mac_out.clone().detach()
                    mac_out_detach[avail_actions == 0] = -9999999
                    cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]

                    if self.args.levin_flag_quantile:
                        cur_max_actions = mac_out_detach[:, 1:].mean(dim=4).max(dim=3, keepdim=True)[1]
                        cur_max_actions = cur_max_actions.unsqueeze(4)
                        cur_max_actions = cur_max_actions.expand(cur_max_actions.size(0), cur_max_actions.size(1), cur_max_actions.size(2), cur_max_actions.size(3), self.args.N_QUANT)
                        
                    target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                else:
                    target_max_qvals = target_mac_out.max(dim=3)[0]    
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]   

        # Mix
        if self.mixer is not None:
        # # levin:  resNet ---- Line2
        #     chosen_action_qvals_levin_res = chosen_action_qvals
        #     target_max_qvals_levin_res = target_max_qvals
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])    # batch['state'] 的维度为 [batch_size, seq_length, state_dim]
            if not self.args.levin_flag_mixer_average:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

            elif self.args.mixer == "qmix":
                if self.args.levin_flag_mix_N_min:
                    for i in range(self.args.average_N_mixer_target):
                        targe_mixer_out = self.target_mixer_list[i](target_max_qvals, batch["state"][:, 1:])
                        if i == 0:
                            targe_mixer_outs = targe_mixer_out
                        else:
                            targe_mixer_outs = th.where(targe_mixer_outs < targe_mixer_out, targe_mixer_outs, targe_mixer_out)
                    target_max_qvals = targe_mixer_outs
                # 使用 mean - std 的方法
                elif self.args.levin_flag_mix_mean_minus_var:
                    for i in range(self.args.average_N_mixer_target):
                        targe_mixer_out = self.target_mixer_list[i](target_max_qvals, batch["state"][:, 1:])
                        if i == 0:
                            cat_tensor_target_mixer_out = targe_mixer_out
                        else:
                            cat_tensor_target_mixer_out = th.cat((cat_tensor_target_mixer_out, targe_mixer_out), 2)
                    mean_tensor = th.mean(cat_tensor_target_mixer_out, 2)
                    std_tensor = th.var(cat_tensor_target_mixer_out, 2)
                    target_max_qvals = (mean_tensor - std_tensor).unsqueeze(2)


                # 求平均
                else:
                    target_max_qvals_sum = 0
                    self.target_mixer_out_list = []
                    for i in range(self.args.average_N_mixer_target):
                        targe_mixer_out = self.target_mixer_list[i](target_max_qvals, batch["state"][:, 1:])
                        target_max_qvals_sum = target_max_qvals_sum + targe_mixer_out
                        if self.args.levin_flag_mixer_select:
                            self.target_mixer_out_list.append(targe_mixer_out)
                    target_max_qvals = target_max_qvals_sum / self.args.average_N_mixer_target



                    # levin: mixer select
                    if self.args.levin_flag_mixer_select:
                        if self.args.levin_name_mixer_select_replacement == 'mean':
                            target_mixer_select_sum = 0
                            for i in range(self.args.average_N_mixer_target):
                                if self.args.levin_flag_mixer_select > 0:
                                    target_mixer_select = th.where(self.target_mixer_out_list[i] < target_max_qvals, target_max_qvals, self.target_mixer_out_list[i])
                                else:
                                    target_mixer_select = th.where(self.target_mixer_out_list[i] > target_max_qvals, target_max_qvals, self.target_mixer_out_list[i])
                                target_mixer_select_sum = target_mixer_select_sum + target_mixer_select
                            target_max_qvals = target_mixer_select_sum / self.args.average_N_mixer_target
                        elif self.args.levin_name_mixer_select_replacement == 'zero':
                            target_mixer_select_sum = 0
                            target_mixer_select_bool_sum = 0
                            for i in range(self.args.average_N_mixer_target):
                                if self.args.levin_flag_mixer_select > 0:
                                    target_mixer_select_bool = (self.target_mixer_out_list[i] > target_max_qvals).float()
                                    target_mixer_select = th.where(self.target_mixer_out_list[i] > target_max_qvals, self.target_mixer_out_list[i], th.full_like(target_max_qvals,0))
                                else:
                                    target_mixer_select_bool = (self.target_mixer_out_list[i] < target_max_qvals).float()
                                    target_mixer_select = th.where(self.target_mixer_out_list[i] < target_max_qvals, self.target_mixer_out_list[i], th.full_like(target_max_qvals,0))
                                target_mixer_select_bool_sum = target_mixer_select_bool_sum + target_mixer_select_bool
                                target_mixer_select_sum = target_mixer_select_sum + target_mixer_select
                            # if self.levin_iter_target_mixer_update < self.args.average_N_mixer_target:
                            if self.levin_iter_target_mixer_update < 2:    # 因为在开始几次，target都是一样的，target_select_bool_sum中只有0和N
                                pass # print("using average-mix directly")
                            else:
                                target_max_qvals = target_mixer_select_sum / target_mixer_select_bool_sum

        # Calculate 1-step Q-Learning targets
        if not self.args.levin_flag_quantile:
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals  # reward和terminated为 [32, xx, 1]
        else:
            targets = rewards.unsqueeze(3) + self.args.gamma * (1 - terminated.unsqueeze(3)) * target_max_qvals

        
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())   

        mask = mask.expand_as(td_error) 
        
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() * 2 
        loss = loss + latent_loss

        
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("loss_levin", loss_levin.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_max_qvals_mean", (target_max_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


    def _update_targets(self):
        if not self.args.levin_flag_average_dqn:
            self.target_mac.load_state(self.mac)
        else:
            self.number = self.levin_iter_target_update % self.args.average_N_target
            self.target_mac_list[self.number].load_state(self.mac)
            self.levin_iter_target_update = self.levin_iter_target_update + 1

        if self.mixer is not None:
            if not self.args.levin_flag_mixer_average:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            elif self.args.mixer == "qmix":
                mixer_number = self.levin_iter_target_mixer_update % self.args.average_N_mixer_target
                self.target_mixer_list[mixer_number].load_state_dict(self.mixer.state_dict())
                self.levin_iter_target_mixer_update = self.levin_iter_target_mixer_update + 1
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        if not self.args.levin_flag_average_dqn:
            self.target_mac.cuda()
        else:
            for i in range(self.args.average_N_target):
                self.target_mac_list[i].cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            if not self.args.levin_flag_mixer_average:
                self.target_mixer.cuda()
            elif self.args.mixer == "qmix":
                for i in range(self.args.average_N_mixer_target):
                    self.target_mixer_list[i].cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        if not self.args.levin_flag_average_dqn:
            self.target_mac.load_models(path)
        else:
            for i in range(self.args.average_N_target):
                self.target_mac_list[i].load_models(path)

        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
