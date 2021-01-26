import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D
import math
import numpy as np


class RNNAgentWithPerception(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentWithPerception, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        origin_input_dim = input_shape
        activation_func=nn.LeakyReLU()

        self.embed_net = nn.Sequential(nn.Linear(input_shape, args.generator_hidden_dim),
                                       nn.BatchNorm1d(args.generator_hidden_dim),
                                       activation_func,
                                       nn.Linear(args.generator_hidden_dim, args.latent_dim * 2))

        self.inference_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + self.state_dim, args.generator_hidden_dim),
                                           nn.BatchNorm1d(args.generator_hidden_dim),
                                           activation_func,
                                           nn.Linear(args.generator_hidden_dim, args.latent_dim * 2))


        # --- confirm dim of input
        if self.args.flag_input_only_latent:
            input_shape = args.latent_dim
        elif self.args.flag_input_only_inputs:
            input_shape = input_shape
        else:
            input_shape = input_shape + args.latent_dim

        if self.args.flag_hyperNet4input:
            input_shape = self.args.dim_hyperNet4input

        if self.args.flag_hyperNet4fc1:
            rnn_hidden_dim_final = self.args.dim_hyperNet4input
        else:
            rnn_hidden_dim_final = args.rnn_hidden_dim       


        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim_final, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)


        # ==== hyper network ====
        if self.args.flag_latent_expand_layer:
            latent_final_dim = args.dim_expand_latent_dim    
        else:
            latent_final_dim = args.latent_dim             

        if self.args.flag_hyperNet4input:
            self.input_w_nn = nn.Linear(latent_final_dim, origin_input_dim * self.args.dim_hyperNet4input)
            self.input_b_nn = nn.Linear(latent_final_dim, self.args.dim_hyperNet4input)
            input_shape = self.args.dim_hyperNet4input
        if self.args.flag_hyperNet4fc1:
            self.input_w_nn = nn.Linear(latent_final_dim, args.rnn_hidden_dim * self.args.dim_hyperNet4input)
            self.input_b_nn = nn.Linear(latent_final_dim, self.args.dim_hyperNet4input)
        if self.args.flag_hyperNet4fc2:
            self.fc2_w_nn = nn.Linear(latent_final_dim, args.rnn_hidden_dim * args.n_actions)
            self.fc2_b_nn = nn.Linear(latent_final_dim, args.n_actions)

        # ===== intention network =====
        self.latent_net = nn.Sequential(nn.Linear(args.latent_dim, args.dim_expand_latent_dim),
                                        nn.BatchNorm1d(args.dim_expand_latent_dim),
                                        activation_func)



    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, state=None):
        mb_size = inputs.size(0)  
        origin_input_dim = inputs.size(1)
        self.bs = int(mb_size / self.n_agents)


        # === latent perception
        latent = self.embed_net(inputs)
        latent[:, self.args.latent_dim:] = th.clamp(th.exp(latent[:, self.args.latent_dim:]), min=self.args.var_floor)
        gaussian_embed = D.Normal(latent[:, :self.args.latent_dim], (latent[:, self.args.latent_dim:]) ** (1 / 2))
        latent = gaussian_embed.rsample()

        # === behavioral intention ===
        if self.args.flag_latent_expand_layer:
            latent_final = self.latent_net(latent)
            hyperNet_input = latent_final
        else:
            latent_final = latent
            hyperNet_input = latent


        # === IAV network ====
        if self.args.flag_input_only_latent:
            inputs_cat_latent = latent
        elif self.args.flag_input_only_inputs:
            inputs_cat_latent = inputs
        else:
            inputs_cat_latent = th.cat((inputs, latent), dim=1)

        if self.args.flag_hyperNet4input:
            input_w = self.input_w_nn(latent_final)
            input_b = self.input_b_nn(latent_final)
            input_w = input_w.view(mb_size, origin_input_dim, -1)  # [mb_size, dim, hyper_dim]
            input_b = input_b.view(mb_size, 1, -1) # [mb_size, 1, hyper_dim]
            
            inputs_unsqz = inputs.unsqueeze(1)   # [mb_size, 1, dim]

            inputs_cat_latent = th.bmm(inputs_unsqz, input_w) + input_b
            inputs_cat_latent = inputs_cat_latent.squeeze(1)

            if self.args.flag_normHyper4input:
                inputs_min=inputs_cat_latent.min(dim=1,keepdim=True)[0]
                inputs_max=inputs_cat_latent.max(dim=1,keepdim=True)[0]

                inputs_cat_latent=(inputs_cat_latent-inputs_min)/(inputs_max-inputs_min+ 1e-12 )

        # ----- fc1 -----
        x = F.relu(self.fc1(inputs_cat_latent))  

        # ----- w_p by hypernetworks ---
        if self.args.flag_hyperNet4fc1:
            input_w = self.input_w_nn(latent_final)
            input_b = self.input_b_nn(latent_final)
            input_w = input_w.view(mb_size, self.args.rnn_hidden_dim, -1)   # [mb_size, hidden_dim, hyper_dim]
            input_b = input_b.view(mb_size, 1, -1)                          # [mb_size, 1, hyper_dim]
            
            x = x.unsqueeze(1)   # [mb_size, 1, dim]

            x = th.bmm(x, input_w) + input_b
            x = x.squeeze(1)

            if self.args.flag_normHyper4input:
                x_min=x.min(dim=1,keepdim=True)[0]
                x_max=x.max(dim=1,keepdim=True)[0]

                x=(x-x_min)/(x_max-x_min+ 1e-12)

        # ----- GRU -----
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)    # 【72,64】
        
        # ------- fc2 ------
        q = self.fc2(h)
        h = h.reshape(-1, self.args.rnn_hidden_dim)
        q = q.view(-1, self.args.n_actions)


        # ==== loss ====
        # ---- perception ----
        if self.args.flag_latent_state_loss:
            state = state.unsqueeze(1).expand(self.bs, self.n_agents, -1).reshape(mb_size,-1)


            latent_state = self.inference_net(th.cat([h_in.detach(), state], dim=1))
            # 裁剪var
            latent_state[:, self.args.latent_dim:] = th.clamp(th.exp(latent_state[:, self.args.latent_dim:]), min=self.args.var_floor)
            gaussian_state = D.Normal(latent_state[:, :self.args.latent_dim], (latent_state[:, self.args.latent_dim:])**(1/2))
            latent_state = gaussian_state.rsample()

            perception_loss = self.args.loss_entropy_weight * gaussian_embed.entropy().sum(dim=-1).mean() + self.args.loss_kl_weight * kl_divergence(gaussian_embed, gaussian_state).sum(dim=-1).mean()

            if self.args.flag_latent_state_loss > 0:
                pass
            else:
                perception_loss = -perception_loss
        else:
            perception_loss = 0

        # ---- agreement loss -----
        if self.args.flag_consensus_loss_pair or self.args.flag_consensus_loss_all:
            latent_dis = latent_final.clone().view(self.bs, self.n_agents, -1)   
            latent_move = latent_final.clone().view(self.bs, self.n_agents, -1)  
            consensus_loss = 0
            for agent_i in range(self.n_agents):
                latent_move = th.cat(
                    [latent_move[:, -1, :].unsqueeze(1), latent_move[:, :-1, :]], dim=1) 
                consensus_loss = consensus_loss + (latent_dis - latent_move)**2 / self.args.latent_dim
            consensus_loss = consensus_loss.sum() / mb_size
        else:
            consensus_loss = 0

        extra_loss = perception_loss + consensus_loss * self.args.consensus_loss_weight

        return q, h, extra_loss