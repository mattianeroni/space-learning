import pygame
import random
import enum
import collections 
import torch
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


# RGB colors 
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREY = (120, 120, 120)


class Game:

    def __init__(self, grid_size=(60, 40), block_size=10, view=2, n_obstacles=100, play_mode=False, speed=40):
        """
        Initialize

        :param grid_size: The size of the world
        :param block_size: The size of a grid cell in the representation 
        :param view: How far the robot sees starting from its position
        :param n_obstacles: The number of obstacles on the grid map 
        :param play_mode: If True the user is playing, otherwise the network
        :param speed: The game speed
        """
        # Informative attributes
        self.window = (grid_size[0] * block_size, grid_size[1] * block_size)
        self.block_size = block_size
        self.view = 2
        self.n_obstacles = n_obstacles
        self.grid_size = grid_size
        self.play_mode = play_mode 
        self.speed = speed
        #self.obstacles = collections.deque()
        
        # Start the pygame frame
        self.display = pygame.display.set_mode(self.window)
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Reset the game 
        self.reset()

        # Human is playing 
        if play_mode:
            self.human_play()


    def reset(self):
        """ Initialize the game """
        self.last_action = torch.tensor([0, 1, 0, 0], dtype=torch.int8) # (UP, DOWN, LEFT, RIGHT)
        self.cpos = (0, 0)
        self.slam = torch.Tensor(self.grid_size[0], self.grid_size[1]).fill_(-1)
        view = self.view
        self.slam[self.cpos] = 0
        self.score = 0
        self._place_obstacles()
        self.frame_iteration = 0


    def _place_obstacles(self):
        """ Place obstacles on the grid map """
        slam, cpos, view = self.slam, self.cpos, self.view
        X, Y = slam.shape
        for _ in range(self.n_obstacles):
            x = random.randint(0, X - 1)
            y = random.randint(0, Y - 1)
            if abs(x - cpos[0]) > view and abs(y - cpos[1]) > view:
                slam[x, y] = 1


    def human_play (self):
        """ A user is playing """
        while True:
            # null action
            action = torch.zeros(4)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = torch.tensor([0, 0, 1, 0])
                    elif event.key == pygame.K_RIGHT:
                        action = torch.tensor([0, 0, 0, 1])
                    elif event.key == pygame.K_UP:
                        action = torch.tensor([1, 0, 0, 0])
                    elif event.key == pygame.K_DOWN:
                        action = torch.tensor([0, 1, 0, 0])

            _, game_over, score = self.play_step(action)
            if game_over:
                break

        pygame.quit()
        print(f"Final Score: {score}")


    def play_step(self, action):
        """ Play a game step as suggested by the network """
        cpos, slam = self.cpos, self.slam

        # 0. Update the game iteration
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        #print(self.slam)
        self._move(action)
        
        # 3. Punish if the robot goes over already visited positions or against obstacles
        reward = 0
        
        if slam[cpos] == 1 or self.out_of_border() or self.frame_iteration > 2000:
            return torch.tensor(-100), True, self.score

        elif slam[cpos] == 0:
            reward = -10
        
        elif slam[cpos] == -1:
            reward = np.where(self.slam.numpy() == 0, 1, 0).sum()
            self.score += 1
        

        # 4. Update SLAM 
        self.slam[cpos] = 0

        # 5. update ui and clock
        self._render()
        self.clock.tick(self.speed)

        # 6. return game over and score
        return torch.tensor(reward), False, self.score


    def out_of_border(self):
        """ Check if the robot is out of border """
        x, y = self.cpos
        X, Y = self.slam.shape 
        return x < 0 or y < 0 or x >= X or y >= Y


    def _render(self):
        """ Render the current state """
        block_size, display = self.block_size, self.display
        X, Y = self.slam.shape
        slam = self.slam 

        # Background
        display.fill(GREY)

        # Plot obstacles, and known vs unknown areas
        for x in range(X):
            for y in range(Y):
                if slam[x, y] == 1:
                    block_x = x * block_size #+ block_size / 2
                    block_y = y * block_size #+ block_size / 2
                    pygame.draw.rect(display, BLACK, pygame.Rect(block_x, block_y, block_size, block_size))
                elif slam[x, y] == 0:
                    block_x = x * block_size #+ block_size / 2
                    block_y = y * block_size #+ block_size / 2
                    pygame.draw.rect(display, WHITE, pygame.Rect(block_x, block_y, block_size, block_size))

        # Plot the robot 
        robot_x = self.cpos[0] * block_size + block_size / 2
        robot_y = self.cpos[1] * block_size + block_size / 2
        pygame.draw.circle(display, RED, (robot_x, robot_y), block_size * 0.35)
        

        text1 = font.render(f"Score: {self.score}", True, BLUE2)
        display.blit(text1, (0, 0))
        text2 = font.render(f"Iter: {self.frame_iteration}", True, BLUE2)
        display.blit(text2, (300, 0))
        pygame.display.flip()


    def _move(self, action):
        self.last_action = action

        # Update robot position
        if action[3] == 1:   # RIGHT
            self.cpos = (self.cpos[0] + 1, self.cpos[1])
        
        elif action[2] == 1:  # LEFT
            self.cpos = (self.cpos[0] - 1, self.cpos[1])

        elif action[1] == 1:  # DOWN
            self.cpos = (self.cpos[0], self.cpos[1] + 1)
        
        elif action[0] == 1:  # UP
            self.cpos = (self.cpos[0], self.cpos[1] - 1)
            


if __name__ == '__main__':
    game = Game(play_mode=True)
