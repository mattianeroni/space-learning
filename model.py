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





class Conv_QNet(nn.Module):

    """ An instance of this class represents a simple  """

    def __init__(self):
        """
        Initialise.

        :param input_size: The number of nodes in input layer
        :param hidden_size: The number of nodes in hidden layer 
        :param output_size: The number of nodes in output layer
        """
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, (3, 3)), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d((2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d((2, 2)))
        self.ln1 = nn.Linear(3328, 16)
        self.relu = nn.ReLU()
        #self.batchnorm = nn.BatchNorm1d(16)
        #self.dropout = nn.Dropout1d(0.5)
        self.ln2 = nn.Linear(16, 5)

        self.ln3 = nn.Linear(10, 256)
        self.ln4 = nn.Linear(256, 5)

        self.ln_final = nn.Linear(10, 4)
        


    def forward(self, grid, x):
        
        if len(grid.shape) == 3:
            grid = grid.unsqueeze(0)

        grid = self.conv1(grid)
        grid = self.conv2(grid)
        #print(grid.shape)
        grid = torch.reshape(grid, (grid.shape[0], -1,))
        #print(grid.shape)
        grid = self.ln1(grid)
        grid = self.relu(grid)
        #grid = self.batchnorm(grid)
        #grid = self.dropout(grid)
        grid = self.ln2(grid)
        grid = self.relu(grid)
        
        x = torch.reshape(x, (grid.shape[0], -1,))
        x = self.ln3(x)
        x = self.relu(x)
        x = self.ln4(x)
        x = self.relu(x)

        x = torch.cat((grid, x), dim=1)
        x = self.relu(x)
        x = self.ln_final(x)
        return x


    def save(self, file_name):
        """ Save the model with current weights into a .pth file """
        torch.save(self.state_dict(), file_name)