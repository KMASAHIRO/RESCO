import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl.nn import Branched
import pfrl.initializers
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead

from .agent import IndependentAgent, Agent

# 方策関数
class OriginalModel(torch.nn.Module):
    def __init__(
        self, num_states, num_actions, num_layers=1, num_hidden_units=128, 
        temperature=1.0, noise=0.0, encoder_type="fc", embedding_type="random", 
        embedding_no_train=True, embedding_num=5, embedding_decay=0.99, beta=0.25, 
        eps=1e-5, device="cpu"):
        
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.temperature = temperature
        self.noise = noise
        self.encoder_type = encoder_type
        self.embedding_type = embedding_type
        self.embedding_no_train = embedding_no_train
        self.embedding_num = embedding_num
        self.embedding_decay = embedding_decay
        self.beta = beta
        self.eps = eps
        self.device = torch.device(device)

        if self.encoder_type == "fc":
            self.fc_first = torch.nn.Linear(self.num_states, num_hidden_units)
            self.encoder = self.fc_encoder
        elif self.encoder_type == "lstm":
            self.lstm = torch.nn.LSTM(self.num_states, num_hidden_units, 1, batch_first=True)
            self.fc_first = torch.nn.Linear(num_hidden_units, num_hidden_units)
            self.encoder = self.lstm_encoder
        elif self.encoder_type == "vq":
            self.fc_first = torch.nn.Linear(self.num_states, num_hidden_units)
            if self.embedding_type == "random":
                embedding = torch.randn(self.embedding_num, num_hidden_units)
            elif self.embedding_type == "one_hot":
                self.embedding_num = num_hidden_units
                embedding = torch.nn.functional.one_hot(torch.tensor(range(num_hidden_units)), num_classes=num_hidden_units)
            
            self.beta_loss_list = list()
            self.middle_outputs = list()
            for i in range(self.embedding_num):
                self.middle_outputs.append(list())
            
            self.embedding = torch.nn.Parameter(embedding, requires_grad=False)
            self.embedding_avg = torch.nn.Parameter(embedding, requires_grad=False)
            self.cluster_size = torch.nn.Parameter(torch.zeros(self.embedding_num), requires_grad=False)
            self.encoder = self.vq_encoder
        
        
        self.fc_actions_layer = torch.nn.Linear(num_hidden_units, self.num_actions)
        self.fc_value_layer = torch.nn.Linear(num_hidden_units, 1)

        fc_layers = list()
        self.num_layers = num_layers
        for i in range(num_layers):
            fc_layers.append(torch.nn.Linear(num_hidden_units, num_hidden_units))
        self.fc_layers = torch.nn.ModuleList(fc_layers)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def fc_encoder(self, inputs):
        x = self.fc_first(inputs)
        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)
            x = self.relu(x)
        
        if self.training:
            x = x + torch.normal(torch.zeros(x.shape), torch.ones(x.shape)*self.noise).to(self.device)
        
        return x
    
    def lstm_encoder(self, inputs):
        if len(inputs.shape) == 2:
            x = inputs.unsqueeze(0)
        _, (x, _) = self.lstm(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_first(x)

        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)
            x = self.relu(x)
        
        if self.training:
            x = x + torch.normal(torch.zeros(x.shape), torch.ones(x.shape)*self.noise).to(self.device)
        
        return x
    
    def vq_encoder(self, inputs):
        x = self.fc_first(inputs)
        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)
            x = self.relu(x)
        
        vector = x.detach()
        if len(vector.shape) == 1:
            embedding_idx = (vector - self.embedding).pow(2).sum(-1).argmin(-1)
        else:
            embedding_idx = (vector.unsqueeze(1) - self.embedding.unsqueeze(0)).pow(2).sum(-1).argmin(-1)

        quantize = x + (self.embedding[embedding_idx] - vector)
        if self.training:
            beta_loss = (x - self.embedding[embedding_idx]).pow(2).mean(-1)
            return quantize, beta_loss, vector, embedding_idx
        else:
            return quantize

    def forward(self, inputs):
        if self.encoder_type == "vq" and self.training:
            x, beta_loss, vector, embedding_idx = self.encoder(inputs)
            if len(beta_loss.shape) != 0 and beta_loss.shape[0] > 1:
                self.beta_loss_list.extend(beta_loss)
                if not self.embedding_no_train:
                    for i in range(len(embedding_idx)):
                        self.middle_outputs[embedding_idx[i]].append(vector[i])
        else:
            x = self.encoder(inputs)

        actions_outputs = self.fc_actions_layer(x)
        actions_prob = self.softmax(actions_outputs/self.temperature)
        value = self.fc_value_layer(x)
        
        return torch.distributions.categorical.Categorical(actions_prob), value
    
    def return_vq_info(self):
        return self.beta, self.beta_loss_list, self.middle_outputs
    
    def reset_vq_info(self):
        self.beta_loss_list = list()
        self.middle_outputs = list()
        for i in range(self.embedding_num):
            self.middle_outputs.append(list())

def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer

# 先行研究での方策関数
class DefaultModel(torch.nn.Module):
    def __init__(
        self, obs_space, act_space, temperature=1.0, noise=0.0, encoder_type=None,
        embedding_type="random", embedding_no_train=True, embedding_num=5, 
        embedding_decay=0.99, beta=0.25, eps=1e-5, device="cpu"):
        
        super().__init__()
        self.temperature = temperature
        self.noise = noise
        self.encoder_type = encoder_type
        self.embedding_type = embedding_type
        self.embedding_no_train = embedding_no_train
        self.embedding_num = embedding_num
        self.embedding_decay = embedding_decay
        self.beta = beta
        self.eps = eps
        self.device = torch.device(device)

        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        h = conv2d_size_out(obs_space[1])
        w = conv2d_size_out(obs_space[2])

        self.conv = lecun_init(nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)))
        self.flatten = nn.Flatten()
        self.linear1 = lecun_init(nn.Linear(h*w*64, 64))
        self.linear2 = lecun_init(nn.Linear(64, 64))
        self.linear4_1 = lecun_init(nn.Linear(64, act_space), 1e-2)
        self.linear4_2 = lecun_init(nn.Linear(64, 1))
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

        if self.encoder_type == "vq":
            if self.embedding_type == "random":
                embedding = torch.randn(self.embedding_num, 64)
            elif self.embedding_type == "one_hot":
                self.embedding_num = 64
                embedding = torch.nn.functional.one_hot(torch.tensor(range(64)), num_classes=64)
            
            self.beta_loss_list = list()
            self.middle_outputs = list()
            for i in range(self.embedding_num):
                self.middle_outputs.append(list())
            
            self.embedding = torch.nn.Parameter(embedding, requires_grad=False)
            self.embedding_avg = torch.nn.Parameter(embedding, requires_grad=False)
            self.cluster_size = torch.nn.Parameter(torch.zeros(self.embedding_num), requires_grad=False)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        if self.noise != 0.0 and self.training:
            x = x + torch.normal(torch.zeros(x.shape), torch.ones(x.shape)*self.noise).to(self.device)
        elif self.encoder_type == "vq":
            vector = x.detach()
            if len(vector.shape) == 1:
                embedding_idx = (vector - self.embedding).pow(2).sum(-1).argmin(-1)
            else:
                embedding_idx = (vector.unsqueeze(1) - self.embedding.unsqueeze(0)).pow(2).sum(-1).argmin(-1)
            
            x = x + (self.embedding[embedding_idx] - vector)
            if self.training:
                beta_loss = (x - self.embedding[embedding_idx]).pow(2).mean(-1)
                if len(beta_loss.shape) != 0 and beta_loss.shape[0] > 1:
                    self.beta_loss_list.extend(beta_loss)
                    if not self.embedding_no_train:
                        for i in range(len(embedding_idx)):
                            self.middle_outputs[embedding_idx[i]].append(vector[i])
        
        actions_outputs = self.linear4_1(x)
        actions_prob = self.softmax(actions_outputs/self.temperature)
        value = self.linear4_2(x)
        
        return torch.distributions.categorical.Categorical(actions_prob), value
    
    def return_vq_info(self):
        return self.beta, self.beta_loss_list, self.middle_outputs
    
    def reset_vq_info(self):
        self.beta_loss_list = list()
        self.middle_outputs = list()
        for i in range(self.embedding_num):
            self.middle_outputs.append(list())

def _elementwise_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: torch.clamp supports clipping to constant intervals
    """
    return torch.min(torch.max(x, x_min), x_max)

class VQ_PPO(PPO):
    def _lossfun(
        self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
        ):
        
        prob_ratio = torch.exp(log_probs - log_probs_old)

        loss_policy = -torch.mean(
            torch.min(
                prob_ratio * advs,
                torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs,
            ),
        )

        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred, vs_teacher)
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred, vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred, vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))

        loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * loss_entropy
        )

        if hasattr(self.model, "encoder_type"):
            if self.model.encoder_type == "vq":
                beta, beta_loss_list, middle_outputs = self.model.return_vq_info()
                vq_loss = 0
                for i in range(len(beta_loss_list)):
                    vq_loss = vq_loss + beta*beta_loss_list[i]
                vq_loss = vq_loss / len(beta_loss_list)

                loss = loss + vq_loss

                self.model.reset_vq_info()

                if not self.model.embedding_no_train:
                    prev_embedding_avg = self.model.embedding_avg.to("cpu")
                    prev_cluster_size = self.model.cluster_size.to("cpu")
                    decay = self.model.embedding_decay
                    embedding_num = self.model.embedding_num
                    eps = self.model.eps

                    chosen_num = list()
                    embedding_sum = list()
                    for i in range(len(middle_outputs)):
                        chosen_num.append(len(middle_outputs[i]))
                        if len(middle_outputs[i]) == 0:
                            embedding_sum.append(torch.zeros(len(prev_embedding_avg[i])))
                        else:
                            embedding_sum.append(torch.stack(middle_outputs[i],dim=0).sum(0).to("cpu"))
                    embedding_avg = decay*prev_embedding_avg + (1-decay)*torch.stack(embedding_sum, dim=0)
                    cluster_size = decay*prev_cluster_size + (1-decay)*torch.tensor(chosen_num)

                    n = cluster_size.sum()
                    cluster_size_norm = (cluster_size + eps) / (n + embedding_num*eps) * n
                    embedding = embedding_avg / cluster_size_norm.unsqueeze(-1)

                    self.model.embedding = torch.nn.Parameter(embedding, requires_grad=False)
                    self.model.embedding_avg = torch.nn.Parameter(embedding_avg, requires_grad=False)
                    self.model.cluster_size = torch.nn.Parameter(cluster_size, requires_grad=False)
                    self.model.to(self.model.device)

        return loss

class IPPO(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number, model_type="default", model_param={}, update_interval=1024, minibatch_size=256, epochs=4, lr=None, decay_rate=None):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]
            self.agents[key] = PFRLPPOAgent(config, obs_space, act_space, model_type, model_param, lr, decay_rate)


class PFRLPPOAgent(Agent):
    def __init__(self, config, obs_space, act_space, model_type="default", model_param={}, update_interval=1024, minibatch_size=256, epochs=4, lr=None, decay_rate=None):
        super().__init__()

        if model_type == "default":
            model_param["obs_space"] = obs_space
            model_param["act_space"] = act_space
            self.model = DefaultModel(**model_param)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        elif model_type == "original":
            num_states = 1
            for dim_num in obs_space:
                num_states *= dim_num
            model_param["num_states"] = num_states
            model_param["num_actions"] = act_space
            
            self.model = OriginalModel(**model_param)

            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': decay_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=lr)
        
        if model_param.get("device"):
            self.device = torch.device(model_param["device"])

        self.agent = VQ_PPO(
            self.model, self.optimizer, gpu=self.device.index,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            clip_eps=0.1,
            clip_eps_vf=None,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            standardize_advantages=True,
            entropy_coef=0.001,
            max_grad_norm=0.5)

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+'.pt')
