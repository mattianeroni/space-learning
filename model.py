import torch
import torch.nn as nn
import torch.nn.functional as activation_function



class Linear_QNet(nn.Module):

    """ An instance of this class represents a simple  """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialise.

        :param input_size: The number of nodes in input layer
        :param hidden_size: The number of nodes in hidden layer 
        :param output_size: The number of nodes in output layer
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass to predict a new action.
        :param x: The input state
        """
        x = activation_function.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name):
        """ Save the model with current weights into a .pth file """
        torch.save(self.state_dict(), file_name)
