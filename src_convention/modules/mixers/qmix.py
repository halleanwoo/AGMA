import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents) 
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        '''
        levin总结：
            首先，将state转化为 [batch_size * seq_length, state_dim]；
            然后，将 satte 输入到 hyper_net 中得到 w 和 b；w和b再经过view得到 三维的。

            agent_qs 也需要转化为 [batch_size × seq_length, 1, action_dim]，然后就可以和 w 、b做运算
        '''
        bs = agent_qs.size(0)
        # import ipdb
        # ipdb.set_trace()
        states = states.reshape(-1, self.state_dim)     # 将[batch_size, seq_length, state_dim] 的state 变为 [batch_size * seq_length, state_dim]
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs 由 [batch_size, seq_len, 5] 变为 [batch_size*seq_len, 1, 5]
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim) # w1: [2048, 5, 32]
        b1 = b1.view(-1, 1, self.embed_dim)             # b1:[2048, 1, 32]
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)       # hidden:[2048, 1, 32]  |||      -- 通过 th.bmm (1,5)*(5,32) = (1, 32) , 将 5个agent的q 合并为 1个，但是是由embed_dim维度表示的
        # import ipdb
        # ipdb.set_trace()
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)   # w_final:[2048, 32, 1]
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)               # v: [2048, 1, 1]
        # Compute final output
        y = th.bmm(hidden, w_final) + v                 # y:[2048, 1, 1]      [2048, 1, 32] * [2048, 32, 1]，将 embed_dim维度 表示为 1 个维度
        # Reshape and return
        q_tot = y.view(bs, -1, 1)                       # q_tot: [32, 64, 1]  即 [32, xx, 1]
        return q_tot


class QMixerQuantilie(nn.Module):
    def __init__(self, args):
        super(QMixerQuantilie, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents * self.args.N_QUANT * self.args.N_QUANT) 
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim * self.args.N_QUANT * self.args.N_QUANT)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim * self.args.N_QUANT)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, self.args.N_QUANT))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents, self.args.N_QUANT)  # agent_qs 由 [32, xx, 5, 200] 变为 [xxxx, 1, 5, 200]
        agent_qs = agent_qs.view(agent_qs.size(0), 1, -1)  # agent_qs 由 [32, xx, 5, 200] 变为 [xxxx, 1, 5*200]

        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents * self.args.N_QUANT, self.embed_dim * self.args.N_QUANT) # w1: [2048, 5*200, 32*200]
        b1 = b1.view(-1, 1, self.embed_dim * self.args.N_QUANT)             # b1:[2048, 1, 32*200]
        # import ipdb
        # ipdb.set_trace()

        hidden = F.elu(th.bmm(agent_qs, w1) + b1)       # hidden:[2048, 1, 32*200]  |||      -- 通过 th.bmm (1,5*200)*(5*200,32*200) = (1, 32*200) , 将 5个agent的q 合并为 1个，但是是由embed_dim维度表示的

        # Second layer
        w_final = th.abs(self.hyper_w_final(states))

        w_final = w_final.view(-1, self.embed_dim * self.args.N_QUANT, self.args.N_QUANT)   # w_final:[2048, 32*200, 1*200]
        # State-dependent bias
        v = self.V(states).view(-1, 1, self.args.N_QUANT)               # v: [2048, 1, 1]
        # Compute final output
        y = th.bmm(hidden, w_final) + v                 # y:[2048, 1, 200]      [2048, 1, 32*200] * [2048, 32*200, 1*200]，将 embed_dim维度 表示为 1 个维度
        # Reshape and return
        q_tot = y.view(bs, -1, 1, self.args.N_QUANT)                       # q_tot: [32, 64, 200]  即 [32, xx, 200]
        return q_tot