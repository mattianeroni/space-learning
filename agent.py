import torch
import random
import numpy as np
import collections
from datetime import datetime 

from model import Linear_QNet, QTrainer
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
        self.memory = collections.deque(maxlen=MAX_MEMORY)
        self.model = model 
        self.trainer = trainer


    def get_state(self):
        game = self.game 
        slam, last_action, cpos = game.slam, game.last_action, game.cpos 
        x, y = cpos

        state = (
            # Current position 
            x, 
            y,

            # Surrounding positions
            slam[x + 1, y],
            slam[x - 1, y],
            slam[x, y + 1],
            slam[x, y - 1]
            
            # Current direction given by last action
            last_action[0],
            last_action[1],
            last_action[2],
            last_action[3],
        )

        return np.array(state, dtype=np.int32)

    def remember(self, state, action, reward, next_state, done):
        """ 
        Store an event in memory.

        NOTE: Memory is a deque, when it is longer than its capacity, the
        first stored events are automatically trashed.
        """
        self.memory.append((state, action, reward, next_state, done)) 
        

    def train_long_memory(self):
        """ Train the model on a batch of events """
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        """ Train the model on a single event """
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
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
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return np.array(final_move, dtype=np.int32)

    
    def train (self):
        """ Train the agent's model """
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0

        while True:
            # Get current state
            state_old = self.get_state()

            # Get the move suggested by the model (with an eventual randomness)
            final_move = self.get_action(state_old)

            # Perform move and get new state
            reward, game_over, score = game.play_step(final_move)
            state_new = self.get_state()

            # Train short memory
            self.train_short_memory(state_old, final_move, reward, state_new, game_over)

            # Save the event in memory
            self.remember(state_old, final_move, reward, state_new, game_over)
            
            # Train long memory and plot results in case of game over
            # or game successfully concluded 
            if game_over:
                game.reset()
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
