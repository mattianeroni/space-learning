import torch
import torch.nn as nn
import torch.optim as optim


class QTrainer:

    """ An instance of this class represents the trainer used to train the model """

    def __init__(self, model, lr=0.001, gamma=0.9):
        """
        Initialise.

        :param model: The model to train 
        :param lr: The learning rate 
        :param gamma: The discount rate 
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()


    def train_step(self, grid, state, action, reward, next_grid, next_state, game_over):
        """
        A training step of the model.

        :param state: Game state before action
        :param action: Action computed
        :param reward: Reward obtained 
        :param game_over: True if the game is finished or game over
        """
        #grid = torch.tensor(grid, dtype=torch.float)
        #state = torch.tensor(state, dtype=torch.float)
        #action = torch.tensor(action, dtype=torch.float)
        #reward = torch.tensor(reward, dtype=torch.float)
        #next_grid = torch.tensor(next_grid, dtype=torch.float)
        #next_state = torch.tensor(next_state, dtype=torch.float)

        # If training is done on only one step, the tensors are rearranged
        if isinstance(game_over, bool):
            grid = torch.unsqueeze(grid, 0)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_grid = torch.unsqueeze(next_grid, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over, )

        # Predict the Q values with current state
        pred = self.model(grid, state)
        target = pred.clone()
        
         
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_grid[idx], next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # Set all parameters to zero
        self.optimizer.zero_grad()

        # Compute the loss on the prediction
        loss = self.criterion(target, pred)

        # Compute derivatives and back propagate
        loss.backward()

        # Compute optimization step adjusting the parameters
        self.optimizer.step()