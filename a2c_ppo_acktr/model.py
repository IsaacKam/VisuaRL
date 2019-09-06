import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, key_value = 0):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if key_value == 1:
           
            base = KeyValueBase
        else:
            if base is None:
                if len(obs_shape) == 3:
                    base = CNNBase
                    # base = RLINE   #Swap CNNBase for RLINE if you want RLINE
                elif len(obs_shape) == 1:
                    base = MLPBase
                else:
                    raise NotImplementedError
        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs, context = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs,context

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _,_ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs,_ = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs, [0,0]

class KeyValueBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(KeyValueBase, self).__init__(recurrent, hidden_size, hidden_size)
        self.hidden_size = hidden_size
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs-3, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
        
        self.kv_extractor = nn.Sequential(
            init_(nn.Conv2d(1, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size*2)), nn.ReLU())


        self.embedding_merge = nn.Sequential(
            init_(nn.Linear(2*hidden_size, 700)), nn.ReLU(),
            init_(nn.Linear(700, hidden_size)), nn.ReLU()
            )


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # K2VS with 3 channels
        x = self.main(inputs[:,:-3,:,:]  / 255.0)
        
        seg1 = self.kv_extractor(inputs[:,-3,:,:].unsqueeze(1)  / 255.0) 
        seg2 = self.kv_extractor(inputs[:,-2,:,:].unsqueeze(1)  / 255.0) 
        seg3 = self.kv_extractor(inputs[:,-1,:,:].unsqueeze(1)  / 255.0)
        keys = torch.cat([s[:,:self.hidden_size].unsqueeze(1) for s in [seg1,seg2,seg3]], 1)
        values = torch.cat([s[:,self.hidden_size:].unsqueeze(1) for s in [seg1,seg2,seg3]], 1) 
        
        # K2VS with 2 channels

        # x = self.main(inputs[:,:-2,:,:]  / 255.0)

        # seg2 = self.kv_extractor(inputs[:,-2,:,:].unsqueeze(1)  / 255.0) 
        # seg3 = self.kv_extractor(inputs[:,-1,:,:].unsqueeze(1)  / 255.0)
        # keys = torch.cat([s[:,:self.hidden_size].unsqueeze(1) for s in [seg2,seg3]], 1)
        # values = torch.cat([s[:,self.hidden_size:].unsqueeze(1) for s in [seg2,seg3]], 1) 
        
        context = keys @ x.unsqueeze(1).transpose(-1,-2)
        alpha = torch.nn.functional.softmax(context, 1)
        context_encoding = (alpha.transpose(-1,-2) @ values).squeeze()
        encoding = self.embedding_merge(torch.cat([x,context_encoding],-1))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(encoding, rnn_hxs, masks)
        
        return self.critic_linear(encoding), encoding, rnn_hxs,alpha.mean(0).squeeze()

class RLINE(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        from torchvision import models
        super(pre_trainedBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        
        original_model = models.vgg19(pretrained=True).cuda()

        class vgg_base(nn.Module):
                    def __init__(self):
                        super(vgg_base, self).__init__()
                        self.features = nn.Sequential(
                            # layer 5
                            # *list(original_model.features.children())[:5]

                            *list(original_model.features.children())[:2] 
                        )
                    def forward(self, x):
                        x = self.features(x)
                        return x

        self.model = vgg_base()

        for param in self.model.parameters():
            param.requires_grad = False

# 5-3-1
        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(64*4, 64*2, 2, stride=3)), nn.ReLU(),
        #     init_(nn.Conv2d(64*2, 64, 2, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(),Flatten(),
        #     init_(nn.Linear(32*5*5, 512)), nn.ReLU())
        
# 5-5-2
        # self.main = nn.Sequential(
        # init_(nn.Conv2d(64*4, 64*3, 4, stride=2)), nn.ReLU(),
        # init_(nn.Conv2d(64*3, 200, 3, stride=2)), nn.ReLU(),
        # init_(nn.Conv2d(200, 150, 3, stride=1)), nn.ReLU(),
        # init_(nn.Conv2d(150, 100, 2, stride=1)), nn.ReLU(),
        # init_(nn.Conv2d(100, 50,2, stride=1)), nn.ReLU() ,Flatten(),
        # init_(nn.Linear(1250, 800)), nn.ReLU(),
        # init_(nn.Linear(800, 512)), nn.ReLU()
        # )

# 5-7-5
        # self.main = nn.Sequential(
        # init_(nn.Conv2d(64*4, 64*3, 4, stride=2)), nn.ReLU(),
        # init_(nn.Conv2d(64*3, 240, 3, stride=1)), nn.ReLU(),
        # init_(nn.Conv2d(240, 200, 3, stride=1)), nn.ReLU(),
        # init_(nn.Conv2d(200, 150, 3, stride=1)), nn.ReLU(),
        # init_(nn.Conv2d(150, 100, 3, stride=1)), nn.ReLU(),
        # init_(nn.Conv2d(100, 50,2, stride=1)), nn.ReLU(), 
        # init_(nn.Conv2d(50, 20,2, stride=1)), nn.ReLU() ,Flatten(),
        # init_(nn.Linear(2000, 1500)), nn.ReLU(),
        # init_(nn.Linear(1500, 1100)), nn.ReLU(),
        # init_(nn.Linear(1100, 900)), nn.ReLU(),
        # init_(nn.Linear(900, 750)), nn.ReLU(),
        # init_(nn.Linear(750, 512)), nn.ReLU())  


# 4-5-2
        self.main = nn.Sequential(
        init_(nn.Conv2d(64*4, 64*3, 5, stride=3)), nn.ReLU(),
        init_(nn.Conv2d(64*3, 200, 4, stride=2)), nn.ReLU(),
        init_(nn.Conv2d(200, 150, 3, stride=1)), nn.ReLU(),
        init_(nn.Conv2d(150, 100, 2, stride=1)), nn.ReLU(),
        init_(nn.Conv2d(100, 50,2, stride=1)), nn.ReLU()
        ,Flatten(),
        init_(nn.Linear(3200, 1600)), nn.ReLU(),
        init_(nn.Linear(1600, 512)), nn.ReLU()
        )



        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        prior1 = self.model(inputs[:,:3,:,:] / 255.0)
        prior2 = self.model(inputs[:,3:6,:,:] / 255.0)
        prior3 = self.model(inputs[:,6:9,:,:] / 255.0)
        prior4 = self.model(inputs[:,9:,:,:] / 255.0)
        
        prior_cats = torch.cat([prior1,prior2,prior3,prior4],1).cuda()
        x = self.main(prior_cats).cuda()


        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
