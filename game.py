import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

#Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREY = (120, 120, 120)

class Game:

    def __init__(self, size=(60,40), n_obstacles=100, block_size=10, speed=40):
        self.w, self.h = size
        self.size = size
        self.n_obstacles = n_obstacles
        self.block_size = block_size
        self.speed = speed
        
        self.display = pygame.display.set_mode((self.w * block_size, self.h * block_size))
        pygame.display.set_caption('Vacuum')
        self.clock = pygame.time.Clock()

        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.grid = torch.full(self.size, -1, dtype=torch.int8)
        self.grid[0, 0] = 0
        self.robot = (0, 0)
        self.score = 0
        self._place_obstacles()
        self.frame_iteration = 0


    def _place_obstacles(self):
        for _ in range(self.n_obstacles):
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            if x != self.robot[0] or y != self.robot[1]:
                self.grid[x, y] = 1


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > self.grid.numel():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        x, y = self.robot 
        if self.grid[x, y] == -1:
            self.grid[x, y] = 0
            self.score += 1
            reward = 10
        elif self.grid[x, y] == 0:
            reward = -1
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, point=None):
        x, y = point or self.robot
        X, Y = self.grid.shape

        if x < 0 or y < 0 or x >= X or y >= Y:
            return True
        
        return self.grid[x, y] == 1


    def is_unknown(self, point=None):
        x, y = point or self.robot

        if self.is_collision((x, y)):
            return False
        
        return self.grid[x, y] == -1


    def _update_ui(self):
        self.display.fill(GREY)

        X, Y = self.grid.shape
        grid, block_size = self.grid, self.block_size
        colors = {1 : BLACK, 0: WHITE}

        for x in range(X):
            for y in range(Y):
                if grid[x, y] != -1:
                    rect = pygame.Rect(x * block_size, y * block_size, block_size, block_size)
                    pygame.draw.rect(self.display, colors[int(grid[x, y])], rect)

        # Plot the robot 
        robot_x = self.robot[0] * block_size + block_size / 2
        robot_y = self.robot[1] * block_size + block_size / 2
        pygame.draw.circle(self.display, RED, (robot_x, robot_y), block_size * 0.35)

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x, y = self.robot
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.robot = (x, y)