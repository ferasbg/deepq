import pygame
from pygame.locals import *
import random

# create grid
# initialize states and actions

class Grid:
    def __init__(self):
        self.grid = [0 for i in range(16)]
        self.state = [0,0,0,0
                      0,0,0,0
                      0,0,0,0]

    # initialize state of all tiles
    # change values in specific tiles given adjacency matches (power of 2)
    # each val in tile is log2(n)
    # display game
    # given specific mouse events, update state_values of tiles (def merge)
    # def empty_tiles (function to check if tiles are empty given dict adjacency checking)


