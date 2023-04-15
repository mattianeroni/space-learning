import torch
import random
import numpy as np
import collections
from datetime import datetime 

from utils import plot



class Agent:

    def __init__(self, game, model, trainer, max_memory=100_000, batch_size=1_000):
        """
        Initialise
        
        :param game: The game or gym
        :param model: The machine learning model
        :param trainer: The training algorithm 
        :param max_memory: The agent memory 
        :param batch_size: The batch size for training 
        
        :attr n_games: The number of games played
        :attr epsilon: The randomness coefficient
        """
        self.n_games = 0
        self.epsilon = 0
        self.memory = collections.deque(maxlen=max_memory)
        self.batch_size = batch_size
        self.game = game
        self.model = model 
        self.trainer = trainer


    def get_state(self):
        game = self.game 
        slam, last_action, cpos = game.slam, game.last_action, game.cpos 
        x, y = cpos
        X, Y = slam.shape
        s0, s1, s2, s3 = 1,1,1,1
        if x + 1 >= 0 and x + 1 < X and y >= 0 and y < Y:
            s0 = slam[x + 1, y]
        if x - 1 >= 0 and x - 1 < X and y >= 0 and y < Y:
            s1 = slam[x - 1, y]
        if x >= 0 and x < X and y + 1 >= 0 and y + 1 < Y:
            s2 = slam[x, y + 1]
        if x >= 0 and x < X and y - 1 >= 0 and y - 1 < Y:
            s3 = slam[x, y - 1]
        return game.slam.unsqueeze(0), torch.tensor([x, y, 
            s0, s1, s2, s3,
            last_action[0], last_action[1], last_action[2], last_action[3]], dtype=torch.float)


    def remember(self, grid, state, action, reward, next_grid, next_state, done):
        """ 
        Store an event in memory.

        NOTE: Memory is a deque, when it is longer than its capacity, the
        first stored events are automatically trashed.
        """
        self.memory.append((grid, state, action, reward, next_grid, next_state, done)) 
        

    def train_long_memory(self):
        """ Train the model on a batch of events """
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        grid, state, action, reward, next_grid, next_state, done = zip(*mini_sample)
        grid = torch.stack(grid)
        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.stack(reward)
        next_grid = torch.stack(next_grid)
        next_state = torch.stack(next_state)
        #print(grid.shape, state.shape)
        self.trainer.train_step(grid, state, action, reward, next_grid, next_state, done)


    def train_short_memory(self, grid, state, action, reward, next_grid, next_state, done):
        """ Train the model on a single event """
        self.trainer.train_step(grid, state, action, reward, next_grid, next_state, done)


    def get_action(self, grid, x):
        """
        Givent a state, this method returns the next action computed by the agent.
        There is a trade-off between exploration and exploitation allowing some 
        random moves during the first itertions.
        """
        self.epsilon = 1.0 / (self.n_games + 1)  #80 - self.n_games
        final_move = [0, 0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            prediction = self.model(grid, x)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return torch.tensor(final_move, dtype=torch.int8)

    
    def train (self):
        """ Train the agent's model """
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0

        while True:
            # Get current state
            grid_old, state_old = self.get_state()

            # Get the move suggested by the model (with an eventual randomness)
            final_move = self.get_action(grid_old, state_old)

            # Perform move and get new state
            reward, game_over, score = self.game.play_step(final_move)
            grid_new, state_new = self.get_state()

            # Train short memory
            self.train_short_memory(grid_old, state_old, final_move, reward, grid_new, state_new, game_over)

            # Save the event in memory
            self.remember(grid_old, state_old, final_move, reward, grid_new, state_new, game_over)
            
            # Train long memory and plot results in case of game over
            # or game successfully concluded 
            if game_over:
                self.game.reset()
                self.n_games += 1
                self.train_long_memory()
                
                if score > record:
                    record = score
                    self.model.save('./model/model.pth')

                _time = datetime.now().strftime("%H:%M:%S.%f")
                print(f"[INFO][{_time}] game: {self.n_games} - score: {score} - record: {record}")

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / self.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
