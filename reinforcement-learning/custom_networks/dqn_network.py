import torch
import torch.nn as nn
import numpy as np

from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

_, nn = try_import_torch()


class DQNNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Arguments:
            - obs_space     : gym.spaces.Space
            - action_space  : gym.spaces.Space
            - num_outputs   : integer
            - model_config  : ModelConfigDict
            - name          : string
        """

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
            )
        nn.Module.__init__(self)

        # size of observation space vector
        self.input_size = model_config['dim']
        self.obs_shape = (1, self.input_size)

        # vector size of each hidden layer
        hiddens = model_config['fcnet_hiddens']

        # the vector size of fully-connected layers
        self.num_outputs = hiddens[-1]

        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_size, hiddens[0]),
            nn.ReLU(),
            nn.Linear(hiddens[0], hiddens[1]),
            nn.ReLU(),
            nn.Linear(hiddens[1], self.num_outputs)
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state=[], seq_lens=None):
        """
        Arguments:
            - input_dict (dict) : dictionary of input tensors, including
                                “obs”, “obs_flat”, “prev_action”, “prev_reward”, “is_training”, “eps_id”, “agent_id”, “infos”, and “t”
            - state (list)      : list of state tensors with sizes matching those returned by get_initial_state + the batch dimension
            - seq_lens (Tensor) : 1d tensor holding input sequence lengths
        Returns:
            - (outputs, state)  : [batch, num_outputs], and the new RNN state.
        """
        
        observation = input_dict["obs"].float()

        # make sure that the shape of the input is as expected
        if observation.shape != self.obs_shape:
            observation = observation[0]
        
        fc_out = self.fc_layers(observation)

        return fc_out, state

    
    def normalize_rgb(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def image_to_dnn_input(self, image):
        # convert width height channel to channel width height
        image = np.array(image.transpose((2, 0, 1)), np.float32)
        # BGRA to BGR
        image = image[:3, :, :]
        # BGR to RGB
        image = image[::-1, :, :]
        # normalize to 0 - 1
        image = image / 255
        # convert image to torch tensor
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        
        # normalize input image (using default torch normalization technique)
        image = self.normalize_rgb(image)

        return image
