"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""

from torch import nn
import torch
def network_factory(in_size, num_actions, env):
    """

    :param in_size:
    :param num_actions:
    :param env: The gym environment. You shouldn't need this, but it's included regardless.
    :return: A network derived from nn.Module
    """
    return PolicyNetwork(in_size, num_actions)

class PolicyNetwork2(nn.Module):
    def __init__(self, in_size, num_actions):
        super(PolicyNetwork2, self).__init__()
        self.p1 = torch.nn.Linear(in_size, 128)
        self.p2 = torch.nn.Linear(128, num_actions)

    def forward(self, inputs):
        # action network
        # print("input: " , inputs.shape)
        x = torch.nn.functional.relu(self.p1(inputs))
        # print("After P1: ", x.shape)
        action_probs = torch.nn.functional.softmax(self.p2(x), -1)
        # print("After P2: ", x.shape)
        # print("Final P: ", action_probs.shape)

        return action_probs, None

    def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
        # print("inputs to the function get_action: ", inputs.shape)
        action_probs, state_value = self.forward(inputs)
        # print("action probs: ", action_probs)
        dist = torch.distributions.Categorical(action_probs) # same as multinomial
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        # self.state_value = state_value
        return action.item(), action_log_prob, state_value

class PolicyNetwork(nn.Module):
    def __init__(self, in_size, num_actions):
        super(PolicyNetwork, self).__init__()
        self.p1 = torch.nn.Linear(in_size, 128)
        self.p2 = torch.nn.Linear(128, num_actions)

        self.v1 = torch.nn.Linear(in_size, 64)
        self.v2 = torch.nn.Linear(64, 1)

    def forward(self, inputs):
        # action network
        # print("input: " , inputs.shape)
        x = torch.nn.functional.relu(self.p1(inputs))
        # print("After P1: ", x.shape)
        action_probs = torch.nn.functional.softmax(self.p2(x), -1)
        # print("After P2: ", x.shape)
        # print("Final P: ", action_probs.shape)

        # state-value network
        x = torch.nn.functional.relu(self.v1(inputs))
        # print("After V1: ", x.shape)

        state_value = self.v2(x)
        # print("After V2: ", state_value.shape)

        return action_probs, state_value

    def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
        # print("inputs to the function get_action: ", inputs.shape)
        action_probs, state_value = self.forward(inputs)
        # print("action probs: ", action_probs)
        dist = torch.distributions.Categorical(action_probs) # same as multinomial
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        # self.state_value = state_value
        return action.item(), action_log_prob, state_value
