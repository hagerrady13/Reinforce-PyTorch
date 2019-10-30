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
"""
Main policy network model
"""
class PolicyNetwork(nn.Module):
    def __init__(self, in_size, num_actions):
        super(PolicyNetwork, self).__init__()
        # action network definition
        self.policy_fc1 = torch.nn.Linear(in_size, 128)
        self.policy_fc2 = torch.nn.Linear(128, num_actions)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

        # value network definition
        self.value_fc1 = torch.nn.Linear(in_size, 64)
        self.value_fc2 = torch.nn.Linear(64, 1)

    def forward(self, inputs):
        # action network
        x = self.relu(self.policy_fc1(inputs))
        action_probs = self.softmax(self.policy_fc2(x))

        # value network
        x = self.relu(self.value_fc1(inputs))
        state_value = self.value_fc2(x)

        return action_probs, state_value

    def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
        action_probs, state_value = self.forward(inputs)

        dist = torch.distributions.Categorical(action_probs) # same as multinomial
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.item(), action_log_prob, state_value

"""
A model defined for the parameterized policy only,
to test whitening without the parameterized state value
"""
class PolicyNetwork2(nn.Module):
    def __init__(self, in_size, num_actions):
        super(PolicyNetwork2, self).__init__()
        self.policy_fc1 = torch.nn.Linear(in_size, 128)
        self.policy_fc2 = torch.nn.Linear(128, num_actions)

    def forward(self, inputs):
        # action network
        x = torch.nn.functional.relu(self.policy_fc1(inputs))
        action_probs = torch.nn.functional.softmax(self.policy_fc2(x), -1)

        return action_probs, None

    def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        """
        action_probs, state_value = self.forward(inputs)
        dist = torch.distributions.Categorical(action_probs) # same as multinomial
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob, state_value
