from agent import Agent 
from model import Linear_QNet
from trainer import QTrainer
from game import Game 


if __name__ == "__main__":
    game = Game(grid_size=(60, 40), block_size=10, view=2, n_obstacles=100, play_mode=False, speed=40)
    model = Linear_QNet(input_size=10, hidden_size=256, output_size=4)
    trainer = QTrainer(model=model, lr=0.001, gamma=0.9) # gamma: discount rate
    
    agent = Agent(game, model, trainer, max_memory=100_000, batch_size=1_000)